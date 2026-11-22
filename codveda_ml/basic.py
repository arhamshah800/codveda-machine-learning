import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import SEED, iris_split, save_figure, validation_folds


def linear_regression(data_dir, output_dir):
    # Codveda's file has a .csv extension but uses whitespace and no header.
    housing = pd.read_csv(data_dir / "housing.csv", sep=r"\s+", header=None)
    if housing.shape[1] != 14:
        raise ValueError("Expected the 14-column Codveda housing dataset.")
    rooms = housing[[5]].rename(columns={5: "average_rooms"})
    value = housing[13]
    x_train, x_test, y_train, y_test = train_test_split(
        rooms, value, test_size=0.2, random_state=SEED
    )
    model = LinearRegression().fit(x_train, y_train)
    baseline = DummyRegressor().fit(x_train, y_train)
    prediction = model.predict(x_test)

    fig, ax = plt.subplots()
    ax.scatter(x_test["average_rooms"], y_test, alpha=0.55, label="Test observations")
    ordered = x_test.sort_values("average_rooms")
    ax.plot(ordered["average_rooms"], model.predict(ordered), color="darkorange", label="Linear fit")
    ax.set(xlabel="Average rooms per dwelling", ylabel="Median home value (historical $1,000s)")
    ax.legend()
    save_figure(fig, output_dir, "fit.png")
    return {
        "r2": r2_score(y_test, prediction),
        "mse_thousand_dollars_squared": mean_squared_error(y_test, prediction),
        "mean_baseline_mse": mean_squared_error(y_test, baseline.predict(x_test)),
        "slope_thousand_dollars_per_room": float(model.coef_[0]),
        "intercept_thousand_dollars": float(model.intercept_),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }


def knn_classifier(data_dir, output_dir):
    x_train, x_test, y_train, y_test = iris_split(data_dir)
    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())
    search = GridSearchCV(
        pipeline, {"kneighborsclassifier__n_neighbors": [1, 3, 5, 7, 9, 11, 15]},
        scoring="accuracy", cv=validation_folds(),
    ).fit(x_train, y_train)
    comparison = pd.DataFrame({
        "k": search.cv_results_["param_kneighborsclassifier__n_neighbors"],
        "mean_cv_accuracy": search.cv_results_["mean_test_score"],
        "std_cv_accuracy": search.cv_results_["std_test_score"],
    })
    comparison.to_csv(output_dir / "k_comparison.csv", index=False)
    predicted = search.predict(x_test)
    display = ConfusionMatrixDisplay.from_predictions(y_test, predicted, colorbar=False)
    display.ax_.set_title("Iris KNN on the test set")
    save_figure(display.figure_, output_dir, "confusion_matrix.png")
    report = classification_report(y_test, predicted, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv", index_label="class")
    return {
        "best_k": int(search.best_params_["kneighborsclassifier__n_neighbors"]),
        "mean_cv_accuracy": search.best_score_,
        "test_accuracy": accuracy_score(y_test, predicted),
        "test_macro_precision": report["macro avg"]["precision"],
        "test_macro_recall": report["macro avg"]["recall"],
    }
