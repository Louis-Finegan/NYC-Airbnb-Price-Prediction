# Model Evaluation Reports

This directory contains evaluation outputs from the Airbnb price prediction project. Numerical results depend on the dataset and model configuration used for each run.

## Files

| File | Contents |
| --- | --- |
| `metrics.json` | Selected model, evaluation metrics, dataset split information and software versions |
| `validation_scores.csv` | Validation results for the candidate models |
| `test_scores.csv` | Test results for the baseline and selected model |
| `holdout_predictions.csv` | Observed prices, predictions and residuals for test listings |
| `split_assignments.csv` | Training, validation and test assignments |
| `predicted_vs_actual.png` | Comparison of predicted and observed prices |
| `residuals.png` | Prediction error diagnostics |

## Evaluation approach

Data is split by host so that each host appears in only one partition. Model selection uses validation mean absolute error in dollars. The selected model and baseline are then evaluated on the same reserved test listings.

## Interpreting the results

- **MAE** measures the average absolute prediction error.
- **RMSE** gives greater weight to large errors.
- **$R^2$** compares prediction errors with a constant prediction based on the evaluation set’s mean price.

Dollar metrics describe errors in nightly prices. Log-price metrics describe errors on a compressed price scale, reducing the influence of very expensive listings.

Read the metrics alongside the diagnostic plots to understand where predictions are less reliable.

## Regenerating reports

Run `python train.py` after preparing the data to regenerate the numerical reports. Run the notebook to reproduce the diagnostic figures.

When updating the analysis, regenerate the related outputs together so they describe the same experiment.