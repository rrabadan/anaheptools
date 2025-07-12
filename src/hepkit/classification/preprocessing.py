import hashlib
from typing import Any
from zlib import crc32

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin


def _validate_split_inputs(
    data_ids: pd.DataFrame, test_ratio: float, id_columns: list[str] | str
) -> list[str]:
    """
    Common validation logic for train/test splitting functions.

    Returns:
        List of validated id_columns
    """
    if not 0.0 <= test_ratio <= 1.0:
        raise ValueError("test_ratio must be between 0.0 and 1.0")

    if isinstance(id_columns, str):
        id_columns = [id_columns]

    missing_cols = [col for col in id_columns if col not in data_ids.columns]
    if missing_cols:
        raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

    return id_columns


def _compute_test_mask(hashes: np.ndarray, test_ratio: float) -> np.ndarray:
    """
    Compute boolean mask for test set based on hash values.

    Returns:
        Boolean array indicating which rows belong to test set
    """
    return hashes < test_ratio * 2**32


def _apply_split(data: pd.DataFrame, test_mask: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the train/test split based on boolean mask.

    Returns:
        Tuple of (train_set, test_set)
    """
    train_set = data.loc[~test_mask].copy().reset_index(drop=True)
    test_set = data.loc[test_mask].copy().reset_index(drop=True)
    return train_set, test_set


def split_train_test_by_id_crc32(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str,
    separator: str = "_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split using CRC32 hash function."""
    id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

    # Create identifier strings
    if len(id_columns) == 1:
        identifiers = data_ids[id_columns[0]].astype(str)
    else:
        identifiers = data_ids[id_columns].apply(
            lambda row: separator.join(row.astype(str)), axis=1
        )

    # Compute CRC32 hashes
    def compute_hash(identifier):
        identifier_bytes = identifier.encode("utf-8")
        return crc32(identifier_bytes) & 0xFFFFFFFF

    hashes = identifiers.apply(compute_hash)
    test_mask = _compute_test_mask(hashes, test_ratio)
    return _apply_split(data, test_mask)


def split_train_test_by_id_hash(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str,
    separator: str = "_",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split using optimized hash functions."""
    id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

    # For single numeric column, use direct numeric hash
    if len(id_columns) == 1 and pd.api.types.is_numeric_dtype(data_ids[id_columns[0]]):
        values = data_ids[id_columns[0]].values
        hashes = (values * 2654435761) % (2**32)  # Knuth's multiplicative hash
    else:
        # String-based approach
        if len(id_columns) == 1:
            identifiers = data_ids[id_columns[0]].astype(str)
        else:
            identifiers = data_ids[id_columns].apply(
                lambda row: separator.join(row.astype(str)), axis=1
            )

        def fast_hash(s):
            return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)

        hashes = identifiers.apply(fast_hash)

    test_mask = _compute_test_mask(hashes, test_ratio)
    return _apply_split(data, test_mask)


def split_train_test_by_id_bitshift(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str = ["runNumber", "eventNumber", "candNumber"],  # noqa: B006
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train/test sets using bit-shifting hash of numeric ID combinations.
    This function creates a deterministic train/test split by combining multiple numeric
    ID columns into a single 64-bit integer using bit-shifting operations, then applying
    a hash function for randomization. The bit-shifting approach is efficient for up to
    3 ID columns with specific range constraints.
    The bit-shifting layout for different column counts:
    - 1 column: Direct use of the column value
    - 2 columns: [run_col (20 bits) | event_col (44 bits)]
    - 3 columns: [run_col (20 bits) | event_col (34 bits) | cand_col (10 bits)]
    For more than 3 columns, automatically falls back to prime multiplication method.
    Parameters
    ----------
    data : pd.DataFrame
        The main dataset to be split.
    data_ids : pd.DataFrame
        DataFrame containing the ID columns used for splitting. Must have the same
        length as `data`.
    test_ratio : float
        Fraction of data to allocate to test set. Must be between 0 and 1.
    id_columns : Union[List[str], str], default ["runNumber", "eventNumber", "candNumber"]
        Column name(s) in `data_ids` to use for creating unique identifiers.
        Can be a single string or list of strings.
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing (train_data, test_data) where both are subsets of the
        original `data` DataFrame.
    Raises
    ------
    ValueError
        If ID column values exceed the bit range limits:
        - 1st column (run): >= 2^20 (1,048,576)
        - 2nd column (event): >= 2^44 for 2-col mode or >= 2^34 for 3-col mode
        - 3rd column (candidate): >= 2^10 (1,024)
    Notes
    -----
    - The split is deterministic: same inputs always produce the same split
    - Uses Knuth's multiplicative hash with constant 2654435761 for randomization
    - Bit-shifting provides better performance than string concatenation methods
    - Range validation ensures no bit overlap in the combined ID
    - For datasets with ID values exceeding range limits, consider using
      `split_train_test_by_id_prime_mult` instead
    Examples
    --------
    >>> data = pd.DataFrame({'feature1': [1, 2, 3, 4], 'target': [0, 1, 0, 1]})
    >>> ids = pd.DataFrame({'run': [1, 1, 2, 2], 'event': [100, 101, 100, 101]})
    >>> train, test = split_train_test_by_id_bitshift(data, ids, 0.5, ['run', 'event'])
    """
    id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

    if len(id_columns) == 1:
        unique_ids = data_ids[id_columns[0]].astype(np.int64)

    elif len(id_columns) == 2:
        run_col, event_col = id_columns[0], id_columns[1]

        # Validate ranges
        max_run = data_ids[run_col].max()
        max_event = data_ids[event_col].max()

        if max_run >= 2**20:
            raise ValueError(
                f"{id_columns[0]} number too large for 2-column mode: {max_run} >= 2^20"
            )
        if max_event >= 2**44:
            raise ValueError(
                f"{id_columns[1]} number too large for 2-column mode: {max_event} >= 2^44"
            )

        unique_ids = (data_ids[run_col].astype(np.int64) << 44) | data_ids[event_col].astype(
            np.int64
        )

    elif len(id_columns) == 3:
        run_col, event_col, cand_col = id_columns[0], id_columns[1], id_columns[2]

        # Validate ranges
        max_run = data_ids[run_col].max()
        max_event = data_ids[event_col].max()
        max_cand = data_ids[cand_col].max()

        if max_run >= 2**20:
            raise ValueError(
                f"{id_columns[0]} number too large for 3-column mode: {max_run} >= 2^20"
            )
        if max_event >= 2**34:
            raise ValueError(
                f"{id_columns[1]} number too large for 3-column mode: {max_event} >= 2^34"
            )
        if max_cand >= 2**10:
            raise ValueError(
                f"{id_columns[2]} number too large for 3-column mode: {max_cand} >= 2^10"
            )

        unique_ids = (
            (data_ids[run_col].astype(np.int64) << 44)
            | (data_ids[event_col].astype(np.int64) << 10)
            | data_ids[cand_col].astype(np.int64)
        )

    else:
        # Fall back to prime method for >3 columns
        return split_train_test_by_id_prime_mult(data, data_ids, test_ratio, id_columns)

    # Apply hash and split
    hashes = (unique_ids * np.int64(2654435761)) % (2**32)
    test_mask = _compute_test_mask(hashes, test_ratio)
    return _apply_split(data, test_mask)


def split_train_test_by_id_prime_mult(
    data: pd.DataFrame, data_ids: pd.DataFrame, test_ratio: float, id_columns: list[str] | str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset into train and test sets using prime multiplication for ID combinations.
    This function creates deterministic train/test splits by combining multiple ID columns
    using prime number multiplication to create unique composite IDs, then hashing these
    IDs to determine the split. This ensures that all rows with the same combination of
    ID values are kept together in either train or test set.
    Args:
        data (pd.DataFrame): The dataset to split.
        data_ids (pd.DataFrame): DataFrame containing ID columns used for splitting.
            Must have the same number of rows as data.
        test_ratio (float): Fraction of unique ID combinations to assign to test set.
            Must be between 0 and 1.
        id_columns (Union[List[str], str]): Column name(s) in data_ids to use for
            creating composite IDs. Can be a single column name or list of column names.
            Maximum of 5 columns supported.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing (train_data, test_data).
    Raises:
        ValueError: If more than 5 ID columns are provided (exceeds available primes).
    Note:
        The function uses predetermined large prime numbers [982451653, 982451629,
        982451581, 982451567, 982451563] to create unique composite IDs. The split
        is deterministic - the same input will always produce the same train/test split.
    Example:
        >>> train, test = split_train_test_by_id_prime_mult(
        ...     data=df,
        ...     data_ids=id_df,
        ...     test_ratio=0.2,
        ...     id_columns=['event_id', 'run_id']
        ... )
    """
    id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

    primes = [982451653, 982451629, 982451581, 982451567, 982451563]

    if len(id_columns) > len(primes):
        raise ValueError(
            f"Too many ID columns ({len(id_columns)}), maximum supported is {len(primes)}"
        )

    # Create unique IDs using prime multiplication
    unique_ids = np.zeros(len(data_ids), dtype=np.int64)
    for i, col in enumerate(id_columns):
        unique_ids += data_ids[col].astype(np.int64) * primes[i]

    # Hash and split
    hashes = (unique_ids * np.int64(2654435761)) % (2**32)
    test_mask = _compute_test_mask(hashes, test_ratio)
    return _apply_split(data, test_mask)


def split_train_test_by_id_polynomial(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str = ["runNumber", "eventNumber", "candNumber"],  # noqa: B006
) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

    if len(id_columns) != 3:
        raise ValueError("Polynomial method requires exactly 3 ID columns")

    BASE = 31  # noqa: N806
    MOD = 2**32 - 1  # noqa: N806

    run_col, event_col, cand_col = id_columns[0], id_columns[1], id_columns[2]

    unique_ids = (
        data_ids[run_col] * BASE * BASE + data_ids[event_col] * BASE + data_ids[cand_col]
    ) % MOD

    hashes = (unique_ids * 2654435761) % (2**32)
    test_mask = _compute_test_mask(hashes, test_ratio)
    return _apply_split(data, test_mask)


# Simplified convenience functions
def split_train_test_by_run_event(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    run_col: str = "runNumber",
    event_col: str = "eventNumber",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function for runNumber + eventNumber splitting."""
    return split_train_test_by_id_bitshift(data, data_ids, test_ratio, [run_col, event_col])


def split_train_test_by_run_event_cand(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    run_col: str = "runNumber",
    event_col: str = "eventNumber",
    cand_col: str = "candNumber",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function for runNumber + eventNumber + candNumber splitting."""
    return split_train_test_by_id_bitshift(
        data, data_ids, test_ratio, [run_col, event_col, cand_col]
    )


def split_train_test_by_unique_id(
    data: pd.DataFrame,
    data_ids: pd.DataFrame,
    test_ratio: float,
    id_columns: list[str] | str = ["runNumber", "eventNumber", "candNumber"],  # noqa: B006
    method: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Unified interface for train/test splitting with automatic method selection.

    Parameters:
    - data: DataFrame containing the dataset to split
    - data_ids: DataFrame containing ID columns for splitting
    - test_ratio: Float between 0.0 and 1.0, proportion of data for test set
    - id_columns: List of column names or single column name to use for ID-based splitting
    - method: "auto" (default) - automatically chooses best method
             "bitshift" - force bit-shifting approach
             "hash" - force hash-based approach
             "crc32" - force CRC32 approach
             "prime" - force prime multiplication approach

    Returns:
    - Tuple of (train_set, test_set) DataFrames
    """
    if method == "auto":
        # id_columns = _validate_split_inputs(data_ids, test_ratio, id_columns)

        # Try bitshift first (fastest), fall back if it fails
        try:
            if len(id_columns) <= 3 and all(
                pd.api.types.is_numeric_dtype(data_ids[col]) for col in id_columns
            ):
                return split_train_test_by_id_bitshift(data, data_ids, test_ratio, id_columns)
        except (ValueError, OverflowError):
            pass

        # Fall back to hash method
        return split_train_test_by_id_hash(data, data_ids, test_ratio, id_columns)

    elif method == "bitshift":
        return split_train_test_by_id_bitshift(data, data_ids, test_ratio, id_columns)
    elif method == "hash":
        return split_train_test_by_id_hash(data, data_ids, test_ratio, id_columns)
    elif method == "crc32":
        return split_train_test_by_id_crc32(data, data_ids, test_ratio, id_columns)
    elif method == "prime":
        return split_train_test_by_id_prime_mult(data, data_ids, test_ratio, id_columns)
    else:
        raise ValueError(
            f"Unknown method: {method}. Choose from: auto, bitshift, hash, crc32, prime"
        )


def _apply_transforms(df: pd.DataFrame, inputvars: list[Any]) -> pd.DataFrame:
    """
    Helper function to apply transformations to input variables.

    Parameters:
    - df: DataFrame containing the raw data
    - inputvars: List of variable objects with name, branch, and expression attributes

    Returns:
    - DataFrame with transformed features
    """
    features = [var.name for var in inputvars]
    transforms = [var.expression for var in inputvars]
    branches = [var.input_branches for var in inputvars]

    transformed_data = {}
    for feature, branch_list, transform in zip(features, branches, transforms, strict=False):
        # Apply transform using the specified branches
        if isinstance(transform, str):
            # If transform is a string, apply identity function (return first branch)
            transformed_data[feature] = df[branch_list[0]]
        else:
            transformed_data[feature] = transform(*(df[b] for b in branch_list))

    return pd.DataFrame(transformed_data)


def prepare_training_data(
    sig_df: pd.DataFrame,
    bkg_df: pd.DataFrame,
    sig_inputvars: list[Any],
    bkg_inputvars: list[Any],
    sig_weights: np.ndarray | None = None,
    bkg_weights: np.ndarray | None = None,
    id_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the data for training by transforming the input variables and adding necessary columns.

    Args:
        sig_df: DataFrame containing the signal data.
        bkg_df: DataFrame containing the background data.
        sig_inputvars: List of input variables for the signal data.
        bkg_inputvars: List of input variables for the background data.
        sig_weights: Array-like object containing the weights for the signal data.
        bkg_weights: Array-like object containing the weights for the background data.
        id_columns: List of ID column names. If None, defaults to standard columns.

    Returns:
        Tuple containing:
        - DataFrame with features, labels, and weights (if provided)
        - DataFrame with ID columns
    """
    if id_columns is None:
        id_columns = ["runNumber", "eventNumber", "candNumber"]

    # Validate ID columns exist
    missing_sig_cols = [col for col in id_columns if col not in sig_df.columns]
    missing_bkg_cols = [col for col in id_columns if col not in bkg_df.columns]
    if missing_sig_cols:
        raise KeyError(f"ID columns missing from signal data: {missing_sig_cols}")
    if missing_bkg_cols:
        raise KeyError(f"ID columns missing from background data: {missing_bkg_cols}")

    # Apply transformations
    sig_transformed = _apply_transforms(sig_df, sig_inputvars)
    bkg_transformed = _apply_transforms(bkg_df, bkg_inputvars)

    # Add ID columns
    for col in id_columns:
        sig_transformed[col] = sig_df[col].values
        bkg_transformed[col] = bkg_df[col].values

    # Add weights if provided
    if sig_weights is not None:
        sig_transformed["weights"] = sig_weights
    if bkg_weights is not None:
        bkg_transformed["weights"] = bkg_weights

    # Add labels
    sig_transformed["label"] = 1
    bkg_transformed["label"] = 0

    # Combine datasets
    combined_data = pd.concat([sig_transformed, bkg_transformed], ignore_index=True)

    # Remove rows with NaN values
    combined_data.dropna(inplace=True)
    combined_data.reset_index(drop=True, inplace=True)

    # Split into features and IDs
    feature_data = combined_data.drop(columns=id_columns)
    id_data = combined_data[id_columns].copy()

    return feature_data, id_data


set_config(transform_output="pandas")


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select specific columns from a DataFrame."""

    def __init__(self, column_names: list[str]):
        """
        Initialize the ColumnSelector.

        Parameters:
        - column_names: List of column names to select
        """
        self.column_names = column_names

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnSelector":  # noqa: N803
        """Fit the transformer (no-op for column selection)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Select the specified columns from the DataFrame."""
        missing_cols = [col for col in self.column_names if col not in X.columns]
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")
        return X[self.column_names].copy()


class ColumnTransformer(BaseEstimator, TransformerMixin):
    """Transform columns using variable expressions."""

    def __init__(self, variables: list[Any], id_columns: list[str] | None = None):
        """
        Initialize the ColumnTransformer.

        Parameters:
        - variables: List of variable objects with name, branch, and expression attributes
        - id_columns: List of ID column names to preserve
        """
        self.variables = variables
        self.id_columns = id_columns or ["runNumber", "eventNumber", "candNumber"]

        # Extract features and transforms
        self.features = [var.name for var in variables]
        self.columns = [
            var.branch if isinstance(var.branch, list) else [var.branch] for var in variables
        ]
        self.transforms = [var.expression for var in variables]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnTransformer":  # noqa: N803
        """Fit the transformer (no-op for this implementation)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Apply transformations to the DataFrame."""
        # Apply transformations
        transformed_data = {}
        for feature, branches, transform in zip(
            self.features, self.columns, self.transforms, strict=False
        ):
            transformed_data[feature] = transform(*(X[b] for b in branches))

        # Create new DataFrame
        result_df = pd.DataFrame(transformed_data)

        # Add ID columns if they exist in input
        for col in self.id_columns:
            if col in X.columns:
                result_df[col] = X[col].values

        # Clean up data
        result_df.dropna(inplace=True)
        result_df.reset_index(drop=True, inplace=True)

        return result_df

    def get_id_columns(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """
        Get ID columns from the last transformed DataFrame.
        This is a separate method to avoid side effects in transform.
        """
        available_id_cols = [col for col in self.id_columns if col in X.columns]
        return X[available_id_cols].copy() if available_id_cols else pd.DataFrame()
