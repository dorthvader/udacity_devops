import pandas as pd
import numpy as np
import pytest
from numpy.random import default_rng
from pathlib import Path
from random import choices
import os
import joblib
from project_1.src.churn_library import train_models
from project_1.src.churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    classification_report_image,
    feature_importance_plot,
    roc_plot,
)

CATAGORICAL_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

INT_COLUMNS = [
    "CLIENTNUM",
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Total_Revolving_Bal",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
]

FLOAT_COLUMNS = [
    "Credit_Limit",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

cols_dict = {
    CATAGORICAL_COLUMNS[0]: choices(["M", "F"], k=50),
    CATAGORICAL_COLUMNS[1]: choices(
        [
            "High School",
            "Graduate",
            "Uneducated",
            "Unknown",
            "College",
            "Post-Graduate",
            "Doctorate",
        ],
        k=50,
    ),
    CATAGORICAL_COLUMNS[2]: choices(["Married", "Single", "Unknown", "Divorced"], k=50),
    CATAGORICAL_COLUMNS[3]: choices(
        [
            "$60K - $80K",
            "Less than $40K",
            "$80K - $120K",
            "$40K - $60K",
            "$120K +",
            "Unknown",
        ],
        k=50,
    ),
    CATAGORICAL_COLUMNS[4]: choices(["Blue", "Gold", "Silver", "Platinum"], k=50),
    "Attrition_Flag": choices(["Existing Customer", "Attrited Customer"], k=50),
}
for key in INT_COLUMNS:
    cols_dict[key] = np.random.randint(50, size=50)

for key in FLOAT_COLUMNS:
    cols_dict[key] = default_rng(42).random((50))

RAW_DF = pd.DataFrame(cols_dict)
CHURN_DF = RAW_DF.copy()
CHURN_DF["Churn"] = CHURN_DF["Attrition_Flag"].apply(
    lambda val: 0 if val == "Existing Customer" else 1
)
CHURN_DF.drop("Attrition_Flag", axis=1, inplace=True)

X_TRAIN = pd.DataFrame({idx: np.random.randint(50, size=35) for idx in range(19)})
X_TEST = pd.DataFrame({idx: np.random.randint(50, size=15) for idx in range(19)})
Y_TRAIN = np.random.randint(2, size=35)
Y_TEST = np.random.randint(2, size=15)

Y_PRED_TRAIN_A = np.random.randint(2, size=35)
Y_PRED_TRAIN_B = np.random.randint(2, size=35)

Y_PRED_TEST_A = np.random.randint(2, size=15)
Y_PRED_TEST_B = np.random.randint(2, size=15)


def test_import_data(tmp_path):
    """Test data import function."""
    csv_path = Path(tmp_path, "raw_df.csv")
    RAW_DF.to_csv(csv_path)

    bad_path_suff = Path(tmp_path, "potato.pdf")
    bad_path_dir = Path(tmp_path, "poatao")

    df = import_data(csv_path)
    assert not df.empty
    assert df.shape[0] > 0
    assert df.shape[1] > 0

    with pytest.raises(FileNotFoundError):
        _ = import_data("potato.csv")
    with pytest.raises(ValueError):
        _ = import_data(bad_path_suff)
        _ = import_data(bad_path_dir)


def test_perform_eda(tmp_path):
    """Test perform eda function."""
    df = perform_eda(df=RAW_DF.copy(), folder_path=tmp_path)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape == RAW_DF.shape

    assert len(os.listdir(tmp_path)) == 5


def test_encoder_helper():
    """Test encoder helper."""
    encoded_df = encoder_helper(df=CHURN_DF.copy(), category_list=CATAGORICAL_COLUMNS)

    assert isinstance(encoded_df, pd.DataFrame)
    assert not encoded_df.empty
    assert encoded_df.shape == CHURN_DF.shape


def test_perform_feature_engineering():
    """Test perform_feature_engineering."""
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=CHURN_DF.copy(), category_list=CATAGORICAL_COLUMNS
    )

    assert X_train.shape[1] == X_test.shape[1] == 19

    assert X_train.shape[0] == y_train.shape[0] == 35
    assert X_test.shape[0] == y_test.shape[0] == 15


def test_train_models(tmp_path):
    """Test train_models."""
    train_models(
        X_train=X_TRAIN,
        X_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST,
        folder_path=tmp_path,
    )

    assert len(os.listdir(tmp_path)) == 2


def test_classification_report_image(tmp_path):
    """Test classification report image."""
    classification_report_image(
        y_train=Y_TRAIN,
        y_test=Y_TEST,
        y_train_preds_lr=Y_PRED_TRAIN_A,
        y_train_preds_rf=Y_PRED_TRAIN_B,
        y_test_preds_lr=Y_PRED_TEST_A,
        y_test_preds_rf=Y_PRED_TEST_B,
        folder_path=tmp_path,
    )

    assert len(os.listdir(tmp_path)) == 2


def test_feature_importance_plot(tmp_path):
    """Test feature_importance_plot."""
    train_models(
        X_train=X_TRAIN,
        X_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST,
        folder_path=tmp_path,
    )

    random_forest_model = joblib.load(Path(tmp_path, "rfc_model.pkl"))

    feature_importance_plot(
        random_forest_model=random_forest_model,
        X_train=X_TRAIN,
        X_test=X_TEST,
        folder_path=tmp_path,
    )

    assert len(os.listdir(tmp_path)) == 4


def test_roc_plot(tmp_path):
    """Test roc plot method."""
    train_models(
        X_train=X_TRAIN,
        X_test=X_TEST,
        y_train=Y_TRAIN,
        y_test=Y_TEST,
        folder_path=tmp_path,
    )

    random_forest_model = joblib.load(Path(tmp_path, "rfc_model.pkl"))
    logistic_regression_model = joblib.load(Path(tmp_path, "logistic_model.pkl"))

    roc_plot(
        random_forest_model=random_forest_model,
        logistic_regression_model=logistic_regression_model,
        X_test=X_TEST,
        y_test=Y_TEST,
        folder_path=tmp_path,
    )

    assert len(os.listdir(tmp_path)) == 3
