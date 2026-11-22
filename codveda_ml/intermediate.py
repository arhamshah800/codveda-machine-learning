import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree

from .common import (
    SEED, binary_metrics, churn_preprocessor, churn_split, iris_split, save_figure, validation_folds,
)


def logistic_regression(data_dir, output_dir):
    x_train, x_test, y_train, y_test = churn_split(data_dir)
    model = make_pipeline(
        churn_preprocessor(scale=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    ).fit(x_train, y_train)

    preprocessing = model.named_steps["columntransformer"]
    coefficients = model.named_steps["logisticregression"].coef_[0]
    odds = pd.DataFrame({
        "feature": preprocessing.get_feature_names_out(),
        "log_odds_coefficient": coefficients,
        "odds_ratio": np.exp(coefficients),
    })
    odds.sort_values("log_odds_coefficient", ascending=False).to_csv(
        output_dir / "odds_ratios.csv", index=False
    )
    encoder = preprocessing.named_transformers_["category"].named_steps["onehotencoder"]
    pd.DataFrame({
        "feature": preprocessing.transformers_[1][2],
        "reference_category": [
            categories[index] for categories, index in zip(encoder.categories_, encoder.drop_idx_)
        ],
    }).to_csv(output_dir / "reference_categories.csv", index=False)

    scores = model.predict_proba(x_test)[:, 1]
    display = RocCurveDisplay.from_predictions(y_test, scores, name="Logistic regression")
    display.ax_.plot([0, 1], [0, 1], linestyle="--", color="gray")
    display.ax_.set_title("Churn logistic regression ROC")
    save_figure(display.figure_, output_dir, "roc_curve.png")
    return binary_metrics(model, x_test, y_test, scores)


def decision_tree(data_dir, output_dir):
    x_train, x_test, y_train, y_test = iris_split(data_dir)
    unpruned = DecisionTreeClassifier(random_state=SEED).fit(x_train, y_train)
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {"ccp_alpha": [0.005, 0.01, 0.02, 0.04, 0.08]},
        scoring="f1_macro", cv=validation_folds(),
    ).fit(x_train, y_train)
    pruned = search.best_estimator_
    pred_base = unpruned.predict(x_test)
    pred_pruned = pruned.predict(x_test)
    pd.DataFrame(search.cv_results_)[
        ["param_ccp_alpha", "mean_test_score", "std_test_score"]
    ].to_csv(output_dir / "pruning_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_tree(
        pruned, feature_names=x_train.columns.tolist(),
        class_names=pruned.classes_.tolist(), filled=True, rounded=True, ax=ax,
    )
    ax.set_title("Iris decision tree after cost-complexity pruning")
    save_figure(fig, output_dir, "pruned_tree.png")
    return {
        "ccp_alpha": search.best_params_["ccp_alpha"],
        "pruned_mean_cv_macro_f1": search.best_score_,
        "unpruned_test_accuracy": accuracy_score(y_test, pred_base),
        "pruned_test_accuracy": accuracy_score(y_test, pred_pruned),
        "unpruned_test_macro_f1": f1_score(y_test, pred_base, average="macro"),
        "pruned_test_macro_f1": f1_score(y_test, pred_pruned, average="macro"),
        "unpruned_leaves": int(unpruned.get_n_leaves()),
        "pruned_leaves": int(pruned.get_n_leaves()),
    }
