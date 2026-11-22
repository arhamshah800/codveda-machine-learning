import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .common import (
    SEED, binary_metrics, churn_preprocessor, churn_split, save_figure, validation_folds,
)


def random_forest(data_dir, output_dir):
    x_train, x_test, y_train, y_test = churn_split(data_dir)
    pipeline = make_pipeline(
        churn_preprocessor(scale=False),
        RandomForestClassifier(random_state=SEED, class_weight="balanced"),
    )
    search = GridSearchCV(
        pipeline,
        {
            "randomforestclassifier__n_estimators": [100, 200],
            "randomforestclassifier__max_depth": [5, 10, None],
            "randomforestclassifier__min_samples_leaf": [1, 4],
        },
        scoring=["precision", "recall", "f1"], refit="f1", cv=validation_folds(), n_jobs=2,
    ).fit(x_train, y_train)
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results[[*cv_results.filter(regex=r"^param_").columns,
                "mean_test_precision", "mean_test_recall", "mean_test_f1", "std_test_f1"]].to_csv(
        output_dir / "cross_validation.csv", index=False
    )

    # Permute each original input column before it enters the pipeline so
    # one-hot columns for the same feature receive one joint importance score.
    importance = permutation_importance(
        search.best_estimator_, x_test, y_test,
        scoring="f1", n_repeats=10, random_state=SEED, n_jobs=2,
    )
    ranked = pd.DataFrame({
        "feature": x_test.columns,
        "mean_f1_drop": importance.importances_mean,
        "std_f1_drop": importance.importances_std,
    }).sort_values("mean_f1_drop", ascending=False)
    ranked.to_csv(output_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    top = ranked.head(10).iloc[::-1]
    ax.barh(top["feature"], top["mean_f1_drop"], xerr=top["std_f1_drop"], color="#37678e")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set(xlabel="Mean drop in test F1 after permutation", title="Random forest feature importance")
    save_figure(fig, output_dir, "feature_importance.png")

    best = search.best_estimator_.named_steps["randomforestclassifier"]
    return {
        "n_estimators": best.n_estimators,
        "max_depth": best.max_depth,
        "min_samples_leaf": best.min_samples_leaf,
        "mean_cv_precision": cv_results.loc[search.best_index_, "mean_test_precision"],
        "mean_cv_recall": cv_results.loc[search.best_index_, "mean_test_recall"],
        "mean_cv_f1": search.best_score_,
        **binary_metrics(search, x_test, y_test, search.predict_proba(x_test)[:, 1]),
    }


def plot_boundaries(parameters, x_train, x_test, y_train, y_test, output_dir):
    columns = ["Total day minutes", "Customer service calls"]
    x_observed = x_test[columns]
    day_grid, call_grid = np.meshgrid(
        np.linspace(0, max(x_train[columns[0]].max(), x_observed[columns[0]].max()), 175),
        np.linspace(0, max(x_train[columns[1]].max(), x_observed[columns[1]].max()), 175),
    )
    grid = pd.DataFrame({columns[0]: day_grid.ravel(), columns[1]: call_grid.ravel()})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for (kernel, params), ax in zip(parameters.items(), axes):
        model = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, class_weight="balanced", C=params["C"], gamma=params["gamma"]),
        ).fit(x_train[columns], y_train)
        score = model.decision_function(grid).reshape(day_grid.shape)
        ax.contourf(day_grid, call_grid, score >= 0, levels=[-0.5, 0.5, 1.5],
                    colors=["#dbeaf5", "#f9e2cc"])
        ax.contour(day_grid, call_grid, score, levels=[0], colors="#263440", linewidths=1)
        for churn, label, color in [(0, "Stayed", "#37678e"), (1, "Churned", "#bd530a")]:
            points = x_observed.loc[y_test == churn]
            ax.scatter(points[columns[0]], points[columns[1]], s=10, alpha=0.55,
                       color=color, label=label)
        ax.set(title=f"{kernel.upper()} kernel", xlabel="Total day minutes")
    axes[0].set_ylabel("Customer service calls")
    axes[1].legend(loc="upper right")
    fig.suptitle("Decision boundaries from SVMs fitted with two features")
    save_figure(fig, output_dir, "decision_boundaries.png")


def support_vector_machine(data_dir, output_dir):
    x_train, x_test, y_train, y_test = churn_split(data_dir)
    results = {}
    parameters = {}
    fig, ax = plt.subplots()
    for kernel, grid in {
        "linear": {"svc__C": [0.1, 1, 10]},
        "rbf": {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", 0.01, 0.1]},
    }.items():
        pipeline = make_pipeline(
            churn_preprocessor(scale=True), SVC(kernel=kernel, class_weight="balanced")
        )
        search = GridSearchCV(
            pipeline, grid, scoring="f1", cv=validation_folds(), n_jobs=2
        ).fit(x_train, y_train)
        best = search.best_estimator_.named_steps["svc"]
        parameters[kernel] = {"C": best.C, "gamma": best.gamma}

        scores = search.decision_function(x_test)
        results[kernel] = {
            **parameters[kernel],
            "mean_cv_f1": search.best_score_,
            **binary_metrics(search, x_test, y_test, scores),
        }
        cv_results = pd.DataFrame(search.cv_results_)
        cv_results[[*cv_results.filter(regex=r"^param_").columns,
                    "mean_test_score", "std_test_score"]].to_csv(
            output_dir / f"{kernel}_cross_validation.csv", index=False
        )
        RocCurveDisplay.from_predictions(y_test, scores, name=kernel.upper(), ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Churn SVM ROC by kernel")
    save_figure(fig, output_dir, "roc_curves.png")
    plot_boundaries(parameters, x_train, x_test, y_train, y_test, output_dir)
    return results
