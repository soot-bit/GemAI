import pandas as pd
from src.loader import load_split


def test_load_split():
    train_df, val_df = load_split()

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)

    assert not train_df.empty
    assert not val_df.empty

    # You can add more specific assertions here, e.g., checking column names or data types
    # For example:


    # The 'x', 'y', 'z' columns are dropped in preprocess_data, and 'volume' is added,
    # but the pickled data might still have them if `preprocess_data` was not applied before pickling.
    # Let's check for the existence of `price_bwp` and some core features.

    assert "price_bwp" in train_df.columns
    assert "carat" in train_df.columns
    assert "cut" in train_df.columns
    assert "price_bwp" in val_df.columns
    assert "carat" in val_df.columns
    assert "cut" in val_df.columns
