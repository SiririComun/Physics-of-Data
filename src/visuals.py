import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

def save_plot(filename, show=True):
    """Saves the plot and optionally displays it in the notebook."""
    path = os.path.join("../artifacts", filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    if show:
        plt.show() # This renders the plot in the notebook
    plt.close()    # This clears the memory
    return path

def plot_categorical_counts(df, columns):
    for col in columns:
        plt.figure()
        sns.countplot(data=df, x=col, hue=col, palette="viridis", legend=False)
        plt.title(f"Distribution of {col}")
        save_plot(f"count_{col}.png")

def plot_numerical_histograms(df, columns):
    for col in columns:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True, color="skyblue")
        plt.title(f"Histogram of {col}")
        save_plot(f"hist_{col}.png")

def plot_boxplot_by_category(df, numerical_col, categorical_col):
    plt.figure()
    sns.boxplot(data=df, x=categorical_col, y=numerical_col, hue=categorical_col, palette="Set2", legend=False)
    plt.title(f"{numerical_col} by {categorical_col}")
    save_plot(f"boxplot_{numerical_col}_{categorical_col}.png")

def plot_scatter_by_category(df, x_col, y_col, hue_col):
    plt.figure()
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, alpha=0.7)
    plt.title(f"Interaction: {x_col} vs {y_col} (Colored by {hue_col})")
    save_plot(f"scatter_{x_col}_{y_col}.png")

def plot_correlation_heatmap(df, method="pearson"):
    """Generates an optimized correlation heatmap for high-dimensional data."""
    plt.figure(figsize=(20, 16)) 
    corr = df.select_dtypes(include=['number']).corr(method=method)
    sns.heatmap( corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 8}, cbar_kws={"shrink": .8})

    # 5. Optimize Labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(f"Correlation Matrix: {method.capitalize()} Coupling", fontsize=16)
    
    return save_plot(f"heatmap_{method}.png")


def plot_overlapping_hist_by_label(df, value_col, label_col, filename, bins=30, alpha=0.55):
    """Creates overlapping histograms for a numeric feature split by categorical labels."""
    palette = {"B": "orange", "M": "blue"}
    plt.figure(figsize=(9, 5))
    for label in ["B", "M"]:
        subset = df[df[label_col] == label][value_col].dropna()
        if subset.empty:
            continue
        plt.hist(
            subset,
            bins=bins,
            alpha=alpha,
            color=palette.get(label, "gray"),
            label=label,
            edgecolor="white"
        )

    plt.title(f"Histogram of {value_col} by {label_col}")
    plt.xlabel(value_col)
    plt.ylabel("Frequency")
    plt.legend(title=label_col)
    return save_plot(filename)


def plot_multifeature_violin(df, feature_columns, label_col, filename):
    """Creates violin plots for multiple numeric features using a long-form melted DataFrame."""
    long_df = pd.melt(
        df,
        id_vars=[label_col],
        value_vars=feature_columns,
        var_name="Feature",
        value_name="Value"
    )

    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=long_df,
        x="Feature",
        y="Value",
        hue=label_col,
        palette={"B": "orange", "M": "blue"},
        cut=0,
        inner="quartile",
        linewidth=1
    )
    plt.title("Multi-feature Violin Plots by Diagnosis")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return save_plot(filename)