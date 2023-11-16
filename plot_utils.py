import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ****************************************************************************
# ****************************************************************************
def display_value_counts_with_order(df, columns, cols_per_row, fig=(20,30)):
    """
    Displays count plots of categorical columns with a specified order of values.

    Parameters:
    - df: Pandas DataFrame.
    - columns (list): List of column names to display.
    - cols_per_row (int): Number of columns to display per row.
    - fig (tuple): Figure size.

    Returns:
    None
    """
    total_size = len(columns)
    nrows = int(np.ceil(len(columns) / cols_per_row))

    fig, axes = plt.subplots(nrows=nrows, ncols=cols_per_row, figsize=fig)
    axes = axes.flatten()
    for ax, column in zip(axes[:total_size], columns[:total_size]):
        top_values = df[column].value_counts().sort_values(ascending=False)
        order = top_values.index
        sns.countplot(x=column, data=df[df[column].isin(top_values.index)], order=order, palette='viridis', ax=ax)

        ax.set_title(column)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_xlabel('')

    plt.tight_layout()
    plt.show();


# ****************************************************************************
# ****************************************************************************
def display_correlation(df, method):
    """
    Displays a heatmap of the correlation matrix of the DataFrame.

    Parameters:
    - df: Pandas DataFrame.
    - method (str): Correlation method ('pearson', 'kendall', 'spearman').

    Returns:
    None
    """
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(method), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show();