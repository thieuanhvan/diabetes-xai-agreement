import logging

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def run_step02_preprocessing(df, config):

    logging.info("============================================================")
    logging.info("STEP 02 - DATA PREPROCESSING")
    logging.info("============================================================")

    target = config["target"]

    X = df.drop(columns=[target])
    y = df[target]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    logging.info(f"Dataset shape: {df.shape}")
    logging.info(f"Number of selected features: {len(X.columns)}")
    logging.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    logging.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y # Van: stratify=y rất quan trọng với dataset imbalance.
    )

    logging.info(f"Train size: {len(X_train)}")
    logging.info(f"Test size : {len(X_test)}")

    data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor
    }

    logging.info("STEP 02 COMPLETED")

    return data