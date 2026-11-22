# Input data

The download bundle includes these files for local reproduction. Git ignores the raw CSVs so they stay out of a new GitHub commit. In a fresh clone, download the four files from the Codveda Drive links below and save them in this folder under the local names shown.

| Local filename | Codveda source | Contents |
| --- | --- | --- |
| `housing.csv` | [4) house Prediction Data Set.csv](https://drive.google.com/file/d/1HZl785L4UV7YR8lVf8GtLvqFiiKxe_hB/view) | 506 rows, whitespace-delimited with 14 columns and no header |
| `iris.csv` | [1) iris.csv](https://drive.google.com/file/d/1MFh6af9VQScrSrw5zUChjqNk3VDD5M0C/view) | 150 rows with four measurements and one species label |
| `churn_train.csv` | [churn-bigml-80.csv](https://drive.google.com/file/d/1YF73QNFjNYNnqqN2vHymT3NQG3qwd4RD/view) | 2,666 training customers, 388 churned |
| `churn_test.csv` | [churn-bigml-20.csv](https://drive.google.com/file/d/1MjXeIXHXj-3LzfsiPkxYKn5VYXeKpopt/view) | 667 test customers, 95 churned |

The [Boston Housing data dictionary](https://herzog.economia.unam.mx/profesores/blopez/valoracion-boston-housing.pdf) identifies column 6 as RM, the average number of rooms per dwelling, and column 14 as MEDV, the median value of owner-occupied homes in thousands of dollars. Each row describes a neighborhood. The regression uses RM as its only predictor, and the fitted values use historical dollar units.

The supplied churn splits have zero identical records in common. All churn models use the supplied training file for fitting and the supplied test file for evaluation. The Iris workflow removes three exact duplicate rows before creating a shared stratified 75/25 split for KNN and the decision tree.
