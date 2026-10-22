# Model results

Selected model: Engineered features + boosting (log, 350).

| Reserved test metric | Original features + random forest | Selected model |
| --- | ---: | ---: |
| MAE_USD | 125.6202 | 97.0330 |
| RMSE_USD | 607.5940 | 572.8579 |
| R2_USD | 0.2022 | 0.2908 |
| MAE_log | 0.3851 | 0.2571 |
| RMSE_log | 0.5292 | 0.3681 |
| R2_log | 0.6073 | 0.8100 |

Mean absolute dollar error fell by 22.76%.

These results use the supplied June 2026 snapshot and 4,641 test listings whose hosts were excluded from development. Selection used validation dollar MAE; the test set did not choose the model. Both models were refitted on the same training + validation hosts.

The saved model is fitted on development hosts only and returns USD per night. All positive finite prices were retained, including expensive listings. The high dollar RMSE and modest dollar R² show that large price errors remain, despite the stronger log-price fit.

The complete notebook executed without errors. Ten regression checks passed, including amenity parsing, no use of monetary target-derived inputs, no host overlap, saved prediction units and downloader failure handling. The live page lookup and a streamed neighbourhood GeoJSON download were also checked.

Validation scores, exact test predictions, row-to-split assignments and software versions are included alongside this file.
