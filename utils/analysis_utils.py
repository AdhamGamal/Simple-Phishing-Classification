# ****************************************************************************
# ****************************************************************************
def check_cardinality(data, mask):
    """
    Checks the cardinality of selected columns in a DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - mask (array-like): A boolean mask indicating which columns to consider.

    Returns:
    pd.DataFrame: A DataFrame containing only the selected columns, with their respective cardinalities printed.
    """
    filtered_columns = data.columns[mask]
    filtered_data = data.loc[:, filtered_columns]
    print(filtered_data.nunique())
    return filtered_data


# ****************************************************************************
# ****************************************************************************
def print_unique_values(data):
    """
    Prints unique values for each column in the DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    None
    """
    for column in data.columns:
        print(f'{column:25s},', data[column].unique().tolist()[:20])


# ****************************************************************************
# ****************************************************************************
def check_negative_count(data):
    """
    Checks and prints the count of negative values in each column of the DataFrame.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    list: A list of column names that contain a large number of negative values.
    """
    columns_to_drop = []

    for column in data.columns:
        negative_count = (data[column] == -1).sum()
        print(f"Negative count in {column:25s}: {negative_count}")
        if negative_count > 5000:
            columns_to_drop.append(column)

    print("\n", f"Columns that contain a large number of negative values: {columns_to_drop}")
    return columns_to_drop


# ****************************************************************************
# ****************************************************************************
def drop_imbalance_features(data):
    """
    Drops columns with imbalanced values based on a threshold.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with columns dropped if their value counts exceed the specified threshold.
    """
    threshold = 0.8
    value_counts = data.apply(lambda x: x.value_counts(normalize=True).max())
    filtered_columns = value_counts[value_counts <= threshold].index
    return data.loc[:, filtered_columns]


# ****************************************************************************
# ****************************************************************************
def drop_highly_correlated_featuers(data):
    """
    Drops columns that are highly correlated based on a threshold, keeping the one with the higher correlation to the target.

    Parameters:
    - data (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with highly correlated columns dropped.
    """
    threshold = 0.8
    correlation_matrix = data.corr()
    highly_correlated_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    for feature1, feature2 in highly_correlated_pairs:
        if feature1 in data.columns and feature2 in data.columns:
            corr_target_feature1 = data[feature1].corr(data['phishing'])
            corr_target_feature2 = data[feature2].corr(data['phishing'])

            if corr_target_feature1 < corr_target_feature2:
                data = data.drop(columns=[feature1])
            else:
                data = data.drop(columns=[feature2])

    return data
