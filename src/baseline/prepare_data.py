"""
Data preparation script.
"""

import pandas as pd

from . import config, constants
from .cleaner import clean_data
from .data_processing import load_and_merge_data
from .features import create_features


def prepare_data() -> None:
    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    merged_df, book_genres_df, _, descriptions_df = load_and_merge_data()

    books_raw = pd.read_csv(config.RAW_DATA_DIR / constants.BOOK_DATA_FILENAME)
    users_raw = pd.read_csv(config.RAW_DATA_DIR / constants.USER_DATA_FILENAME)
    clean_df, book_mapping = clean_data(merged_df, books_raw, users_raw)

    book_genres_df[constants.COL_BOOK_ID] = (
        book_genres_df[constants.COL_BOOK_ID].map(book_mapping).fillna(book_genres_df[constants.COL_BOOK_ID])
    )
    book_genres_df = book_genres_df.drop_duplicates()
    descriptions_df[constants.COL_BOOK_ID] = (
        descriptions_df[constants.COL_BOOK_ID].map(book_mapping).fillna(descriptions_df[constants.COL_BOOK_ID])
    )
    descriptions_df["len"] = descriptions_df[constants.COL_DESCRIPTION].str.len()
    descriptions_df = descriptions_df.sort_values("len", ascending=False).drop_duplicates(
        subset=constants.COL_BOOK_ID, keep="first"
    )
    descriptions_df = descriptions_df.drop(columns=["len"])

    featured_df = create_features(clean_df, book_genres_df, descriptions_df, include_aggregates=False)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    print(f"\nSaving processed data to {processed_path}...")
    featured_df.to_parquet(processed_path, index=False, engine="pyarrow", compression="snappy")
    train_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN])
    test_rows = len(featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST])

    print("\nData preparation complete!")
    print(f"  - Train rows: {train_rows:,}")
    print(f"  - Test rows: {test_rows:,}")
    print(f"  - Output file: {processed_path}")


if __name__ == "__main__":
    prepare_data()
