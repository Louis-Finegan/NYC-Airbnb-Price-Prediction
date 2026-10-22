"""Compare price models on grouped validation data and evaluate one final choice."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder
from threadpoolctl import threadpool_limits

from clean import DEFAULT_OUTPUT, prepare_data
from features import BASE_CATEGORICAL, CATEGORICAL, make_features

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/airbnb_price_model.pkl"
REPORT_DIR = ROOT / "reports"
SEED = 42


def split_data(frame):
    """Separate hosts into roughly 60% train, 20% validation and 20% test."""
    if frame["host_id"].isna().any() or frame["host_id"].nunique() < 5:
        raise ValueError("At least five nonmissing host groups are needed.")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    development, test = next(outer.split(frame, groups=frame["host_id"]))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train, validation = next(inner.split(frame.iloc[development],
                                        groups=frame.iloc[development]["host_id"]))
    splits = {"train": development[train], "validation": development[validation],
              "test": test}
    host_sets = [set(frame.iloc[index]["host_id"]) for index in splits.values()]
    if any(host_sets[i] & host_sets[j] for i in range(3) for j in range(i)):
        raise AssertionError("A host occurs in more than one partition.")
    return splits


def build_model(feature_set="engineered", algorithm="boosting", target="log",
                **parameters):
    """Return a complete pipeline whose predictions are always dollars/night."""
    categories = BASE_CATEGORICAL if feature_set == "baseline" else CATEGORICAL
    numeric_columns = make_column_selector(dtype_exclude=object)
    if algorithm == "forest":
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
             numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categories),
        ])
        estimator = RandomForestRegressor(
            n_estimators=300, min_samples_split=5, random_state=SEED, n_jobs=-1,
            **parameters,
        )
    elif algorithm == "boosting":
        # Categories are kept nominal. Unknown categories encode as missing (-1).
        preprocessor = ColumnTransformer([
            ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                           unknown_value=-1, max_categories=200), categories),
            ("numeric", "passthrough", numeric_columns),
        ], sparse_threshold=0)
        settings = dict(max_iter=350, learning_rate=0.06, max_leaf_nodes=31,
                        min_samples_leaf=25, l2_regularization=5.0,
                        early_stopping=False, categorical_features=list(range(len(categories))),
                        random_state=SEED)
        settings.update(parameters)
        estimator = HistGradientBoostingRegressor(**settings)
    else:
        raise ValueError("algorithm must be 'forest' or 'boosting'.")
    pipeline = Pipeline([
        ("features", FunctionTransformer(make_features, validate=False,
                                         kw_args={"feature_set": feature_set})),
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])
    if target == "log":
        return TransformedTargetRegressor(regressor=pipeline, func=np.log1p,
                                          inverse_func=np.expm1)
    if target == "dollars":
        return TransformedTargetRegressor(regressor=pipeline)
    raise ValueError("target must be 'log' or 'dollars'.")


def candidate_models():
    """A small, fixed comparison; choose using validation dollar MAE only."""
    return {
        "Original features + random forest": build_model("baseline", "forest"),
        "Engineered features + random forest": build_model("engineered", "forest"),
        "Engineered features + boosting (log, 350)": build_model(),
        "Engineered features + boosting (log, 600)": build_model(
            max_iter=600, max_leaf_nodes=20, min_samples_leaf=35,
            l2_regularization=10.0),
        "Engineered features + boosting (dollars, Poisson)": build_model(
            target="dollars", loss="poisson", max_iter=500,
            max_leaf_nodes=20, min_samples_leaf=30, l2_regularization=10.0),
    }


def score_predictions(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Predictions must be finite, nonnegative dollar prices.")
    actual_log, predicted_log = np.log1p(actual), np.log1p(predicted)
    return {
        "MAE_USD": float(mean_absolute_error(actual, predicted)),
        "RMSE_USD": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2_USD": float(r2_score(actual, predicted)),
        "MAE_log": float(mean_absolute_error(actual_log, predicted_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(actual_log, predicted_log))),
        "R2_log": float(r2_score(actual_log, predicted_log)),
    }


def run_experiment(frame, model_path=MODEL_PATH, report_dir=REPORT_DIR):
    """Select on validation; refit on development; evaluate the reserved test set.

    The saved model is fitted on development hosts only, so the reported test
    errors apply to that exact model. The test set never selects a candidate.
    """
    if not np.isfinite(frame["price"]).all() or not frame["price"].gt(0).all():
        raise ValueError("Run clean.py first: price must be positive and finite.")
    inputs = frame.drop(columns=["price", "id", "host_id"], errors="ignore")
    prices = frame["price"]
    splits = split_data(frame)
    train, validation, test = (splits[name] for name in ["train", "validation", "test"])
    candidates = candidate_models()
    rows = []
    with threadpool_limits(limits=4):
        for name, model in candidates.items():
            start = time.perf_counter()
            model.fit(inputs.iloc[train], prices.iloc[train])
            metrics = score_predictions(prices.iloc[validation], model.predict(inputs.iloc[validation]))
            rows.append({"model": name, **metrics,
                         "fit_seconds": round(time.perf_counter() - start, 2)})
            print(f"{name}: validation MAE ${metrics['MAE_USD']:.2f}; "
                  f"log RMSE {metrics['RMSE_log']:.3f}", flush=True)
        validation_scores = pd.DataFrame(rows).set_index("model").sort_values(
            "MAE_USD", kind="stable")
        selected = validation_scores.index[0]
        baseline_name = next(iter(candidates))
        development = np.sort(np.concatenate([train, validation]))
        final_model = clone(candidates[selected])
        final_model.fit(inputs.iloc[development], prices.iloc[development])
        baseline = clone(candidates[baseline_name])
        if selected == baseline_name:
            baseline = final_model
        else:
            baseline.fit(inputs.iloc[development], prices.iloc[development])
        predictions = final_model.predict(inputs.iloc[test])
        baseline_predictions = baseline.predict(inputs.iloc[test])
        test_scores = pd.DataFrame.from_dict({
            "Baseline": score_predictions(prices.iloc[test], baseline_predictions),
            "Selected": score_predictions(prices.iloc[test], predictions),
        }, orient="index")
        development_scores = score_predictions(prices.iloc[development],
                                               final_model.predict(inputs.iloc[development]))

    split_info = pd.DataFrame([
        {"partition": name, "listings": len(index),
         "hosts": frame.iloc[index]["host_id"].nunique()}
        for name, index in splits.items()
    ]).set_index("partition")
    holdout = frame.iloc[test][["id", "host_id", "neighbourhood_group_cleansed",
                               "room_type", "price"]].copy()
    holdout["baseline_prediction"] = baseline_predictions
    holdout["prediction"] = predictions
    holdout["residual"] = holdout["price"] - holdout["prediction"]
    metadata = {
        "selected_model": selected,
        "selection_metric": "validation MAE_USD (lower is better)",
        "split_method": "GroupShuffleSplit by host_id; seeds 42 and 43",
        "split": split_info.reset_index().to_dict(orient="records"),
        "validation_scores": validation_scores.reset_index().to_dict(orient="records"),
        "test_scores": test_scores.to_dict(orient="index"),
        "development_scores": development_scores,
        "price_range_USD": [float(prices.min()), float(prices.max())],
        "target_rows_removed_or_capped_for_outliers": 0,
        "prediction_units": "USD per night; predict() already inverts any log transform",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
    }
    model_path, report_dir = Path(model_path), Path(report_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path, compress=3)
    (report_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validation_scores.to_csv(report_dir / "validation_scores.csv")
    test_scores.to_csv(report_dir / "test_scores.csv")
    holdout.to_csv(report_dir / "holdout_predictions.csv", index=False)
    assignments = frame[["id", "host_id"]].copy()
    assignments["partition"] = ""
    for name, index in splits.items():
        assignments.iloc[index, assignments.columns.get_loc("partition")] = name
    assignments.to_csv(report_dir / "split_assignments.csv", index=False)
    print(f"\nSelected: {selected}\n{test_scores.round(3).to_string()}", flush=True)
    print(f"Saved model to {model_path}", flush=True)
    return {"model": final_model, "selected": selected, "validation": validation_scores,
            "test": test_scores, "holdout": holdout, "split_info": split_info,
            "splits": splits, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--reports", type=Path, default=REPORT_DIR)
    args = parser.parse_args()
    if not args.data.exists() and args.data == DEFAULT_OUTPUT:
        prepare_data()
    frame = pd.read_csv(args.data, low_memory=False)
    run_experiment(frame, args.model, args.reports)


if __name__ == "__main__":
    main()
