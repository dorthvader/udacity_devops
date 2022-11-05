"""Udacity Project 1: Customer Churn"""
import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split


def main(
    data_path: Union[str, Path],
    catagorical_columns: list,
    dependent_input: Optional[str] = None,
    dependent_var: Optional[str] = None,
):
    """Train Random Forest/Logistic Regression Models; graph their performance and underlying data.

    Input:
        data_path: path to raw csv (only one file allowed)
        catagorical_columns: list of columns to treat as catagorical
        dependent_input: name of columns used to generate dependent variable
        dependent_var: name to use for generated dependent variable.
    """
    logging.info(
        "Beginning main using the following arguments\n    data_path: %s\n   "
        " catagorical_columns: %s\n    dependent_input: %s\n    dependent_var: %s",
        data_path,
        catagorical_columns,
        dependent_input,
        dependent_var,
    )
    raw_df = import_data(data_path)
    df = perform_eda(
        raw_df, dependent_input=dependent_input, dependent_var=dependent_var
    )
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=df, category_list=catagorical_columns, dependent_var=dependent_var
    )
    train_models(X_train, X_test, y_train, y_test)
    logging.info("This program is complete - thank you")


def import_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Returns dataframe for the csv found at data_path."""
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    if data_path.is_dir():
        logging.error(
            "The data file path provided must be to a single csv. A folder path was"
            " provided."
        )
        raise ValueError
    if data_path.suffixes[-1] != ".csv":
        logging.error(
            "The data file path provided must be to a single csv. Other file types are"
            " not supported."
        )
        raise ValueError
    try:
        raw_df = pd.read_csv(data_path, index_col=0)
        logging.info("Successfully loaded file: %s", data_path)
        return raw_df
    except PermissionError as err:
        logging.error(
            "Access denied for file: %s\n Try Updating File Permissions\n%s",
            data_path,
            err,
        )
        raise
    except FileNotFoundError as err:
        logging.error("File not found at %s\n%s", data_path, err)
        raise


def perform_eda(
    df: pd.DataFrame,
    folder_path: Optional[str] = "images/eda",
    dependent_input: Optional[str] = "Attrition_Flag",
    dependent_var: Optional[str] = "Churn",
) -> pd.DataFrame:
    """Create dependent variable and save exploratory images to folder_path."""
    logging.info("Beggining exploratory data analysis.")
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    df[dependent_var] = df[dependent_input].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    df.drop(dependent_input, axis=1, inplace=True)

    plt.figure(figsize=(20, 10))
    df[dependent_var].hist()
    plt.title("Churn Distribution")
    plt.xlabel("0: Remaining Customer 1: Churn")
    plt.ylabel("Total Churn Counts")
    plt.savefig(Path(folder_path, "churn_distribution.png"))
    plt.clf()

    df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Customer Age in Years")
    plt.ylabel("Customer Age Counts")
    plt.savefig(Path(folder_path, "customer_age_distribution.png"))
    plt.clf()

    df.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.xticks(rotation=0)
    plt.ylabel("Counts of Customers by Marital Status")
    plt.savefig(Path(folder_path, "marital_status_distribution.png"))
    plt.clf()

    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total Numer of Transactions Distribution")
    plt.xlabel("Number of Total Transactions by Individual")
    plt.ylabel("Count of Customers by Number of Total Transactions")
    plt.savefig(Path(folder_path, "total_transaction_distribution.png"))
    plt.clf()

    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.xticks(rotation=60)
    plt.title("Heatmap")
    plt.tight_layout()
    plt.savefig(Path(folder_path, "heatmap.png"))
    plt.clf()

    return df


def encoder_helper(
    df: pd.DataFrame, category_list: List[str], dependent_var: Optional[str] = "Churn"
) -> pd.DataFrame:
    """Replace each categorical column with propotion of dependent var per unique value.
    """
    groups = df[category_list].apply(
        lambda col: df.groupby(col).mean(numeric_only=True)[dependent_var]
    )

    for column in category_list:
        df[column + "_" + dependent_var] = df[column].apply(
            lambda sample: groups.loc[
                sample, column  # pylint: disable="cell-var-from-loop"
            ]
        )

    df.drop(category_list, axis=1, inplace=True)
    return df


def perform_feature_engineering(
    df: pd.DataFrame,
    category_list: List[str],
    drop_cols: Optional[list] = None,
    dependent_var: Optional[str] = "Churn",
) -> tuple:
    """Create train test split on data.
    input:
              df: pandas dataframe
              category_list: list of categorical variables
              drop_cols: list of columns to drop - should not contain dependent variable
              dependent_var: columns selected as the dependent variable

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    logging.info("Beginning feature engineering.")

    if drop_cols is None:
        drop_cols = ["CLIENTNUM"]

    df = encoder_helper(df=df, category_list=category_list, dependent_var=dependent_var)
    y = df[dependent_var]
    drop_cols.append(dependent_var)
    X = df.drop(drop_cols, axis=1)
    return train_test_split(X, y, test_size=0.3, random_state=42)


def train_models(
    X_train,
    X_test,
    y_train: npt.ArrayLike,
    y_test: npt.ArrayLike,
    folder_path: str = "./models",
):
    """Train logistic regression and random forest models; save best models to model_path.

    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    random_forest_classifier = RandomForestClassifier(random_state=42)

    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    logistic_regression_classifier = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    grid_search_random_forest = GridSearchCV(
        estimator=random_forest_classifier, param_grid=param_grid, cv=5
    )
    logging.info("Starting gridsearch for random forest classifier.")
    grid_search_random_forest.fit(X_train, y_train)

    logging.info("Training logistic regression classifier.")
    logistic_regression_classifier.fit(X_train, y_train)

    y_train_preds_rf = grid_search_random_forest.best_estimator_.predict(X_train)
    y_test_preds_rf = grid_search_random_forest.best_estimator_.predict(X_test)

    y_train_preds_lr = logistic_regression_classifier.predict(X_train)
    y_test_preds_lr = logistic_regression_classifier.predict(X_test)

    logging.info(
        "Saving trained models to %s", folder_path
    )  # ideally a try except block would be used here
    joblib.dump(
        grid_search_random_forest.best_estimator_, Path(folder_path, "rfc_model.pkl")
    )
    joblib.dump(logistic_regression_classifier, Path(folder_path, "logistic_model.pkl"))

    logging.info("Creating and saving ROC plot.")
    roc_plot(
        random_forest_model=grid_search_random_forest.best_estimator_,
        logistic_regression_model=logistic_regression_classifier,
        X_test=X_test,
        y_test=y_test,
    )

    logging.info("Creating and saving classificaiton report images.")
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )

    logging.info("Creating and saving feature importance plots.")
    feature_importance_plot(
        grid_search_random_forest.best_estimator_, X_train=X_train, X_test=X_test
    )


def roc_plot(
    random_forest_model,
    logistic_regression_model,
    X_test,
    y_test,
    folder_path: str = "images/results",
) -> None:
    """Produce and save roc plot."""
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    lrc_plot = plot_roc_curve(logistic_regression_model, X_test, y_test)
    plt.figure(figsize=(15, 8))
    plt_ax = plt.gca()
    plot_roc_curve(random_forest_model, X_test, y_test, ax=plt_ax, alpha=0.8)
    lrc_plot.plot(ax=plt_ax, alpha=0.8)
    plt.savefig(Path(folder_path, "roc_curve_result.png"))
    plt.clf()


def classification_report_image(
    y_train: npt.ArrayLike,
    y_test: npt.ArrayLike,
    y_train_preds_lr: npt.ArrayLike,
    y_train_preds_rf: npt.ArrayLike,
    y_test_preds_lr: npt.ArrayLike,
    y_test_preds_rf: npt.ArrayLike,
    folder_path: str = "images/results",
):  # pylint: disable="too-many-arguments"
    """Produce and save classification report for training and testing results.
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    forest_test = classification_report(
        y_test, y_test_preds_rf, target_names=["No Churn", "Churn"]
    )
    forest_train = classification_report(
        y_train, y_train_preds_rf, target_names=["No Churn", "Churn"]
    )

    log_test = classification_report(
        y_test, y_test_preds_lr, target_names=["No Churn", "Churn"]
    )
    log_train = classification_report(
        y_train, y_train_preds_lr, target_names=["No Churn", "Churn"]
    )

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01, 0.05, str(forest_train), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.35,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(0.01, 0.7, str(forest_test), {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")
    plt.savefig(Path(folder_path, "random_forest_results.png"))
    plt.clf()

    plt.rc("figure", figsize=(5, 5))
    plt.text(
        0.01,
        1,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(0.01, 0.05, str(log_train), {"fontsize": 10}, fontproperties="monospace")
    plt.text(
        0.01,
        0.35,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(0.01, 0.7, str(log_test), {"fontsize": 10}, fontproperties="monospace")
    plt.axis("off")
    plt.savefig(Path(folder_path, "logistic_results.png"))
    plt.clf()


def feature_importance_plot(
    random_forest_model, X_train, X_test, folder_path: str = "images/results"
):
    """Creates and stores feature importances in output_pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    X = pd.concat([X_train, X_test])
    # Calculate feature importances
    importances = random_forest_model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X.shape[1]), names, rotation=60)
    plt.tight_layout()
    plt.savefig(Path(folder_path, "feature_importances.png"))
    plt.clf()

    explainer = shap.TreeExplainer(random_forest_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(Path(folder_path, "logistic_regression_classification_report.png"))


if __name__ == "__main__":
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename="./logs/churn_library.log",
        level=logging.INFO,
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
    )

    CAT_COLUMNS = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    DEP_INPUT = "Attrition_Flag"
    DEP_VAR = "Churn"

    parser = ArgumentParser()
    parser.add_argument("--data-path", type=str, help="Location of data csv.")
    parser.add_argument(
        "--dependent-column",
        type=str,
        help="Column to use to create dependent variable.",
    )
    parser.add_argument(
        "--dependent-name",
        type=str,
        help="Desired name of generated dependent variable.",
    )
    parser.add_argument(
        "--catagorical_columns",
        type=list,
        help="List of columns that should be considered catagorical.",
    )
    args = parser.parse_args()

    if args.catagorical_columns:
        CAT_COLUMNS = args.catagorical_columns
    if args.dependent_column:
        DEP_INPUT = args.dependent_column
    if args.dependent_name:
        DEP_VAR = args.dependent_name

    main(
        data_path=args.data_path,
        catagorical_columns=CAT_COLUMNS,
        dependent_input=DEP_INPUT,
        dependent_var=DEP_VAR,
    )
