import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .advanced import random_forest, support_vector_machine
from .basic import knn_classifier, linear_regression
from .intermediate import decision_tree, logistic_regression

TASKS = {
    "linear_regression": (linear_regression, ("housing.csv",)),
    "knn": (knn_classifier, ("iris.csv",)),
    "logistic_regression": (logistic_regression, ("churn_train.csv", "churn_test.csv")),
    "decision_tree": (decision_tree, ("iris.csv",)),
    "random_forest": (random_forest, ("churn_train.csv", "churn_test.csv")),
    "svm": (support_vector_machine, ("churn_train.csv", "churn_test.csv")),
}


def main():
    parser = argparse.ArgumentParser(description="Run the Codveda machine learning experiments.")
    parser.add_argument("task", nargs="?", choices=["all", *TASKS], default="all")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    selected = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    required = sorted({filename for _, files in selected.values() for filename in files})
    missing = [str(args.data_dir / name) for name in required if not (args.data_dir / name).is_file()]
    if missing:
        parser.error(f"Missing input files: {', '.join(missing)}. See data/README.md.")

    for name, (run, _) in selected.items():
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        metrics = run(args.data_dir, folder)
        path = folder / "metrics.json"
        path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()
