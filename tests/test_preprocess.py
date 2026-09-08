"""Tests for src/preprocess.py - cleaning and splitting behavior."""

import pandas as pd

from src.preprocess import (
    fix_invalid_categories,
    remove_duplicates,
    split_features_target,
    split_train_test,
)


def test_remove_duplicates_drops_exact_dupes(sample_raw_df):
    df_with_dupe = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]], ignore_index=True)
    assert len(df_with_dupe) == len(sample_raw_df) + 1

    result = remove_duplicates(df_with_dupe)
    assert len(result) == len(sample_raw_df)


def test_fix_invalid_categories_remaps_education(sample_raw_df_with_dirty_categories):
    result = fix_invalid_categories(sample_raw_df_with_dirty_categories)
    assert result["EDUCATION"].isin([1, 2, 3, 4]).all()


def test_fix_invalid_categories_remaps_marriage(sample_raw_df_with_dirty_categories):
    result = fix_invalid_categories(sample_raw_df_with_dirty_categories)
    assert result["MARRIAGE"].isin([1, 2, 3]).all()


def test_split_features_target_separates_correctly(sample_raw_df, sample_target):
    df = sample_raw_df.copy()
    df["default"] = sample_target

    X, y = split_features_target(df)
    assert "default" not in X.columns
    assert list(y) == list(sample_target)


def test_split_features_target_missing_target_raises(sample_raw_df):
    try:
        split_features_target(sample_raw_df, target_col="default")
        assert False, "Expected KeyError for missing target column"
    except KeyError:
        pass


def test_split_train_test_shapes(sample_raw_df, sample_target):
    # Use a slightly bigger synthetic set so stratified split has enough
    # samples per class - 3 rows is too small for a real stratified split.
    X = pd.concat([sample_raw_df] * 10, ignore_index=True)
    y = pd.concat([sample_target] * 10, ignore_index=True)

    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.3, random_state=42)

    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_train) == len(y_train)