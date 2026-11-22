# Codveda Machine Learning Tasks

Six experiments for the Codveda Technology machine learning internship, implemented in Python with pandas and scikit-learn. Each task writes its metrics, figures, and supporting tables to its own folder.

| Level | Codveda task | Dataset | Outputs |
| --- | --- | --- | --- |
| Basic | 2 · Simple linear regression | Housing | [Test metrics and fitted line](results/linear_regression/) |
| Basic | 3 · K-nearest neighbors | Iris | [K comparison, confusion matrix, and per-class metrics](results/knn/) |
| Intermediate | 1 · Logistic regression | Telecom churn | [ROC curve, odds ratios, and category references](results/logistic_regression/) |
| Intermediate | 2 · Decision tree | Iris | [Pruning scores and tree visualization](results/decision_tree/) |
| Advanced | 1 · Random forest | Telecom churn | [Cross-validation metrics and feature importance](results/random_forest/) |
| Advanced | 2 · Linear and RBF SVMs | Telecom churn | [Kernel comparison, ROC curves, and decision boundaries](results/svm/) |

## Run locally

Use Python 3.12. From the repository root, create a virtual environment and install the pinned dependencies.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If you cloned the repository, get the four input files using the links and filenames in [data/README.md](data/README.md). The download bundle already includes them.

```bash
python -m codveda_ml
python -m codveda_ml --help
python -m codveda_ml svm --output-dir rerun-results
```

The default command runs all six tasks. The optional task name runs a single experiment, and `--data-dir` and `--output-dir` let you use different input and output folders.

## Evaluation design

- Housing uses a seeded 80/20 split. The mean baseline is trained on the same data as the fitted line.
- Both Iris classifiers use the same stratified 75/25 split. KNN selects the number of neighbors by accuracy. The tree selects a positive cost-complexity penalty by macro F1. Both searches use five stratified folds on the training partition.
- Churn models share Codveda's supplied train and test partitions. The training pipelines fit imputation, encoding, and any scaling inside the cross-validation folds used for tuning. Area code is handled as a category. Usage charges are derived from minutes, so the models use minutes as the input signal.
- All churn models weight classes inversely to their training frequency. Random forest and SVM hyperparameters are selected by five-fold cross-validated F1 for the churn class. The test set has 95 churners out of 667 customers, so a classifier predicting that everyone stays would score 85.8% accuracy with zero churn recall.
- The two SVM boundary plots use separate two-feature fits so both axes have interpretable units. The reported kernel metrics and ROC curves come from the full-feature fits.

## Test results

### Housing and Iris

| Experiment | Test result | Interpretation |
| --- | --- | --- |
| Simple linear regression | R² 0.371, MSE 46.145 in squared thousands of dollars | The training fit associates one additional average room per dwelling with $9,348 higher predicted neighborhood median value. The mean baseline had MSE 75.045. The negative intercept is an extrapolation to zero rooms. |
| KNN | 91.9% accuracy, 93.8% macro precision, 91.7% macro recall | Five neighbors won the training cross-validation and correctly classified 34 of 37 test flowers. |
| Pruned decision tree | 89.2% accuracy, 0.892 macro F1 | With `ccp_alpha=0.02`, pruning reduced the tree from nine leaves to four and preserved the test scores of the unpruned tree. |

### Telecom churn

| Model | Accuracy | Churn precision | Churn recall | Churn F1 | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.774 | 0.361 | 0.768 | 0.492 | 0.819 |
| Random forest | 0.927 | 0.717 | 0.800 | 0.756 | 0.922 |
| Linear SVM | 0.778 | 0.367 | 0.768 | 0.497 | 0.821 |
| RBF SVM | 0.876 | 0.546 | 0.747 | 0.631 | 0.913 |

The [logistic coefficient table](results/logistic_regression/odds_ratios.csv) reports exp(coefficient) for each input. For example, holding the other predictors fixed, having an international plan multiplies the model's fitted odds of churn by 11.68 relative to having no international plan. A numeric odds ratio applies to a one-training-standard-deviation increase in that feature. The [category reference table](results/logistic_regression/reference_categories.csv) lists the baseline for each categorical input. These are associations estimated by a class-weighted model.

The random forest achieved the highest test F1 and correctly identified 76 of the 95 churners. Permuting daytime minutes dropped its test F1 by 0.349 on average, followed by customer-service calls at 0.196. The [feature importance figure](results/random_forest/feature_importance.png) shows how much the fitted model depends on those features on the supplied test set. The bars show the standard deviation across ten shuffles.

The RBF kernel improved SVM test F1 from 0.497 to 0.631. Its precision increased from 0.367 to 0.546, while recall fell from 0.768 to 0.747. The [decision boundary figure](results/svm/decision_boundaries.png) illustrates how the kernels divide the feature plane when each model is trained on only daytime minutes and customer-service calls.
