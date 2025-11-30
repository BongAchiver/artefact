"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model.
Updates: Handles deduplicated IDs via 'book_id_original' and supports CatBoost.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import config, constants
from .features import add_aggregate_features, handle_missing_values


def predict() -> None:
    """Generates and saves predictions for the test set.

    This script loads prepared data from data/processed/, computes aggregate features
    on all train data, applies them to test set, and generates predictions using
    the trained model.

    It maps predictions back to the RAW test file structure to ensure
    no pairs are missing in the submission.
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print(f"Train set: {len(train_set):,} rows")
    print(f"Test set: {len(test_set):,} rows")

    # Compute aggregate features on ALL train data (to use for test predictions)
    print("\nComputing aggregate features on all train data...")
    test_set_with_agg = add_aggregate_features(test_set.copy(), train_set)

    # Handle missing values (use train_set for fill values)
    print("Handling missing values...")
    test_set_final = handle_missing_values(test_set_with_agg, train_set)

    # Load trained model to get feature definitions
    # Check for the specific CatBoost model file first, fallback to config filename
    model_path = config.MODEL_DIR / "catboost_model.cbm"

    if not model_path.exists():
        model_path = config.MODEL_DIR / config.MODEL_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading model from {model_path}...")
    model = CatBoostRegressor()
    model.load_model(str(model_path))

    print("Aligning test data with model features...")
    expected_features = model.feature_names_

    missing_cols = set(expected_features) - set(test_set_final.columns)
    if missing_cols:
        print(f"Warning: filling missing columns {missing_cols} with zeros")
        for c in missing_cols:
            test_set_final[c] = 0

    X_test = test_set_final[expected_features].copy()
    cat_indices = model.get_cat_feature_indices()
    if len(cat_indices) > 0:
        for idx in cat_indices:
            col_name = expected_features[idx]
            X_test[col_name] = X_test[col_name].astype(str)

    predict_pool = Pool(data=X_test, cat_features=cat_indices)

    # Generate predictions
    print("Generating predictions...")
    test_preds = model.predict(predict_pool)
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)
    test_set_final[constants.COL_PREDICTION] = clipped_preds

    print("\n=== ДЕЛАЕМ МАППИНГ ===")
    raw_test = pd.read_csv(config.RAW_DATA_DIR / constants.TEST_FILENAME)
    print(f"СТРОК В ТЕСТЕ: {len(raw_test)}")

    if "book_id_original" in test_set_final.columns:
        print("book_id_original НАЙДЕН")
        preds_map = test_set_final[[constants.COL_USER_ID, "book_id_original", constants.COL_PREDICTION]].copy()
        preds_map = preds_map.rename(columns={"book_id_original": constants.COL_BOOK_ID})
    else:
        print("book_id_original НЕ НАЙДЕН, ТРАБЛ В cleaner.py")
        preds_map = test_set_final[[constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_PREDICTION]].copy()

    preds_map = preds_map.drop_duplicates(subset=[constants.COL_USER_ID, constants.COL_BOOK_ID])
    submission_df = raw_test.merge(preds_map, on=[constants.COL_USER_ID, constants.COL_BOOK_ID], how="left")
    missing_mask = submission_df[constants.COL_PREDICTION].isna()
    if missing_mask.sum() > 0:
        global_mean = train_set[config.TARGET].mean()
        submission_df.loc[missing_mask, constants.COL_PREDICTION] = global_mean

    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"Сабмит: {submission_path}")
    print(
        f"Predictions: min={submission_df[constants.COL_PREDICTION].min():.4f}, "
        f"max={submission_df[constants.COL_PREDICTION].max():.4f}, "
        f"mean={submission_df[constants.COL_PREDICTION].mean():.4f}"
    )


if __name__ == "__main__":
    predict()
