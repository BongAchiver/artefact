import pandas as pd

from . import constants


def clean_data(df: pd.DataFrame, books_df: pd.DataFrame, users_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Суть очистки:
    1. Чиним аномалии в возрасте (5 < x < 100 норм аудитория, остальное рофл юзеры я думаю).
    2. Разные издания одной книги считаем за одну и ту же
    """
    print("=" * 40)
    print("ОЧИСТКА ДАННЫХ")
    print("=" * 40)

    if "book_id_original" not in df.columns:
        df["book_id_original"] = df[constants.COL_BOOK_ID].copy()

    print("=== ЧИСТКА ЮЗЕРОВ ===")
    median_age = users_df["age"].median()
    bad_age_mask = (users_df["age"] > 100) | (users_df["age"] < 5)
    print(f"  Fixed {bad_age_mask.sum()} user ages.")
    users_df.loc[bad_age_mask, "age"] = median_age
    cols_to_drop = [c for c in users_df.columns if c in df.columns and c != constants.COL_USER_ID]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    df = df.merge(users_df, on=constants.COL_USER_ID, how="left")

    print("=== ЧИСТИМ ИЗДАНИЯ ===")
    books_df["title_clean"] = books_df["title"].astype(str).str.lower().str.strip()
    books_df["author_clean"] = books_df["author_name"].astype(str).str.lower().str.strip()
    books_df["canonical_id"] = books_df.groupby(["title_clean", "author_clean"])[constants.COL_BOOK_ID].transform("min")

    mapping = dict(zip(books_df[constants.COL_BOOK_ID], books_df["canonical_id"]))
    n_total = len(books_df)
    n_unique = books_df["canonical_id"].nunique()
    print(f"НАШЛИ {n_total - n_unique} ДУБЛИКАТОВ")

    original_ids = df[constants.COL_BOOK_ID].copy()
    df[constants.COL_BOOK_ID] = df[constants.COL_BOOK_ID].map(mapping).fillna(df[constants.COL_BOOK_ID])
    remapped_count = (original_ids != df[constants.COL_BOOK_ID]).sum()
    print(f"РЕМАПНУТО {remapped_count}")

    clean_books_df = books_df.drop_duplicates(subset="canonical_id").copy()
    clean_books_df = clean_books_df.drop(columns=["title_clean", "author_clean", "canonical_id"])
    cols_to_drop = [c for c in clean_books_df.columns if c in df.columns and c != constants.COL_BOOK_ID]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    df = df.merge(clean_books_df, on=constants.COL_BOOK_ID, how="left")
    return df, mapping
