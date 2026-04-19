import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy.stats import norm

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


def plot_overlapping_pdfs_and_histograms(
    mu1,
    sigma1,
    mu2,
    sigma2,
    samples_state_0,
    samples_state_1,
    boundary,
    filename="03_phase_boundary.png",
):
    """Plots Gaussian PDFs and sampled histograms, marking the analytical phase boundary."""
    min_x = min(np.min(samples_state_0), np.min(samples_state_1), mu1 - 4 * sigma1, mu2 - 4 * sigma2)
    max_x = max(np.max(samples_state_0), np.max(samples_state_1), mu1 + 4 * sigma1, mu2 + 4 * sigma2)
    x_grid = np.linspace(min_x, max_x, 1200)

    pdf_0 = norm.pdf(x_grid, loc=mu1, scale=sigma1)
    pdf_1 = norm.pdf(x_grid, loc=mu2, scale=sigma2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_grid, pdf_0, color="#2a9d8f", lw=2, label="State 0 PDF")
    axes[0].plot(x_grid, pdf_1, color="#e76f51", lw=2, label="State 1 PDF")
    axes[0].axvline(boundary, color="black", lw=2, ls="--", label=f"x* = {boundary:.3f}")
    axes[0].set_title("Overlapping Probability Densities")
    axes[0].set_xlabel("X_1")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(samples_state_0, bins=25, density=True, alpha=0.58, color="#2a9d8f", label="Label 0")
    axes[1].hist(samples_state_1, bins=25, density=True, alpha=0.58, color="#e76f51", label="Label 1")
    axes[1].axvline(boundary, color="black", lw=2, ls="--", label=f"x* = {boundary:.3f}")
    axes[1].set_title("Overlapping Training Histograms")
    axes[1].set_xlabel("X_1")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.tight_layout()
    return save_plot(filename)


def plot_decision_boundary_2d(model, x_values, y_values, filename, title="Decision Boundary"):
    """Plots 2D decision regions and observed samples for binary labels."""
    if x_values.shape[1] != 2:
        raise ValueError("x_values must have exactly 2 features for 2D boundary plotting.")

    x0_min, x0_max = x_values[:, 0].min() - 1.0, x_values[:, 0].max() + 1.0
    x1_min, x1_max = x_values[:, 1].min() - 1.0, x_values[:, 1].max() + 1.0

    grid_x0, grid_x1 = np.meshgrid(
        np.linspace(x0_min, x0_max, 300),
        np.linspace(x1_min, x1_max, 300),
    )
    grid_points = np.c_[grid_x0.ravel(), grid_x1.ravel()]
    region_pred = model.predict(grid_points).reshape(grid_x0.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(grid_x0, grid_x1, region_pred, alpha=0.25, cmap="coolwarm")
    plt.scatter(
        x_values[:, 0],
        x_values[:, 1],
        c=y_values,
        cmap="coolwarm",
        edgecolor="white",
        alpha=0.9,
        s=40,
    )
    plt.xlabel("X_1")
    plt.ylabel("X_2")
    plt.title(title)
    return save_plot(filename)