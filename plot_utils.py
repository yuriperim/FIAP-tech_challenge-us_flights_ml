import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns


def topk_barplot(pd_series: pd.Series, normalize: bool = True, k: int = 10, group_others: bool = True) -> Axes:
    freq_count = pd_series.value_counts(normalize=normalize)

    if group_others and (len(freq_count) > k):
        data = pd.concat([
            freq_count.head(k),
            pd.Series({"Other": freq_count.iloc[k:].sum()}),
        ])
    else:
        data = freq_count.head(k)

    ax = data.plot.bar(figsize=(8, 5))

    ax.set_xlabel("Count")
    ax.set_ylabel(pd_series.name)

    return ax


def cumsum_plot(pd_series: pd.Series, annotate: bool = True, threshold: float = 0.8, ax: Axes | None = None) -> Axes:
    freq_count = pd_series.value_counts(normalize=True).cumsum()

    x = np.arange(len(freq_count) + 1) / len(freq_count)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        x[1:],
        freq_count.values,
        linestyle="-",
        color="black",
        label=pd_series.name,
    )

    ax.plot(
        x,
        x,
        linestyle="--",
        color="blue",
        label="Uniform",
    )

    alpha = 1.1609  # 80/20 rule (~log_4(5))
    ax.plot(
        x,
        x ** (1 - 1 / alpha),
        linestyle="-.",
        color="blue",
        label='"Pareto"',
    )

    if annotate:
        idx_threshold = np.searchsorted(freq_count.values, threshold)
        x_threshold = x[idx_threshold + 1]
        y_threshold = freq_count.values[idx_threshold]

        ax.axvline(
            x=x_threshold,
            linestyle=":",
            color="red",
        )

        ax.axhline(
            y=y_threshold,
            linestyle=":",
            color="red",
        )

        ax.scatter(
            x=x_threshold,
            y=y_threshold,
            color="red",
        )

        ax.text(
            x=x_threshold + 0.025,
            y=y_threshold - 0.125,
            s=(
                f"{y_threshold:.2%} occurrencies\n"
                "explained by\n"
                f"{idx_threshold + 1} categories ({x_threshold:.2%})"
            ),
            color="red",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)

    ax.set_xlabel("Fraction of Categories")
    ax.set_ylabel("Cumulative Frequency")
    ax.set_title(pd_series.name)
    ax.legend()

    return ax


def plot_cm(cm: np.ndarray, model_name: str) -> None:
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted No Delay", "Predicted Delay"],
        yticklabels=["Actual No Delay", "Actual Delay"]
    )
    plt.title(f"Confusion Matrix — {model_name}")
    plt.show()
