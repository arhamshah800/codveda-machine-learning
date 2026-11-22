import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42


def validation_folds():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def iris_split(data_dir):
    iris = pd.read_csv(data_dir / "iris.csv").drop_duplicates()
    features = iris.drop(columns="species")
    species = iris["species"]
    return train_test_split(
        features, species, test_size=0.25, stratify=species, random_state=SEED
    )


def churn_split(data_dir):
    train = pd.read_csv(data_dir / "churn_train.csv")
    test = pd.read_csv(data_dir / "churn_test.csv")
    # The charge columns are derived from minutes. Use minutes to avoid
    # redundant signals and to make importance scores easier to interpret.
    charges = train.filter(regex=r"^Total .* charge$").columns.tolist()
    x_train = train.drop(columns=["Churn", *charges]).astype({"Area code": str})
    x_test = test.drop(columns=["Churn", *charges]).astype({"Area code": str})
    return x_train, x_test, train["Churn"].astype(int), test["Churn"].astype(int)


def churn_preprocessor(scale):
    numeric_steps = [SimpleImputer(strategy="median")]
    if scale:
        numeric_steps.append(StandardScaler())
    return ColumnTransformer([
        ("numeric", make_pipeline(*numeric_steps), make_column_selector(dtype_include="number")),
        ("category", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
        ), make_column_selector(dtype_exclude="number")),
    ])


def binary_metrics(model, x_test, y_test, scores):
    predicted = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted, zero_division=0),
        "recall": recall_score(y_test, predicted, zero_division=0),
        "f1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
    }


def save_figure(fig, output_dir, name):
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
