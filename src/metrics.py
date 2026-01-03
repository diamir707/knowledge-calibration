from typing import (
    List,
    Literal,
    Tuple,
    Optional,
    Union, Any)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame
from sklearn.calibration import calibration_curve


def adaptive_calibration_error(
        predictions: List[int],
        confidences: List[float],
        n_bins: int = 20,
        norm: Literal["l1", "l2"] = "l1"
        ) -> float:
    """
    Variant of the expected calibration error with adaptive bins, i.e.
    where each bin contains approximately the same number of samples.
    Modified function adopted from Nixon et al. (2019):
    https://github.com/JeremyNixon/uncertainty-metrics-1
    ---------------
    :param predictions: 1d array with the binaries if a prediction was correct/incorrect.
    :param confidences: 1d array with associated confidences.
    :param n_bins: int, number of bins.
    :param norm: str, the norm applied to the errors.
    :returns: float, the adaptive calibration error.
    """

    binned_correct, binned_confs = calibration_curve(
        y_true=predictions,
        y_prob=confidences,
        n_bins=n_bins,
        strategy="quantile"
    )
    errors = np.abs(binned_correct-binned_confs)
    if norm == "l1":
        errors = errors
    elif norm == "l2":
        errors = np.square(errors)
    else:
        raise ValueError(f"Unknown norm: {norm}")
    return np.mean(errors)


def weighted_average(groups: pd.DataFrame,
                     predictions: str
                     ) -> float:
    """
    Implements the weighted average over all relations for a given column of
    correctness labels.
    ---------------
    :param groups: the per-relation grouped dataframe.
    :param predictions: the column name of per-relation average of instance-level correctness.
    :returns: the weighted average over all relations.
    """
    weights = groups["instance"]
    values = groups[predictions]
    return round((values*weights).sum()/weights.sum(), 6)


def brier_score(predictions: List[int],
                confidences: List[float]
                ) -> float:
    """Function which implements the brier score."""
    bs = np.sum((np.array(predictions) - np.array(confidences))**2)*(1/len(predictions))
    return bs


def plot_calibration_curve(
        title: str,
        predictions: List[List[int]],
        confidences: List[List[float]],
        n_bins: int = 20,
        binning_strategy: Literal["uniform", "quantile"] = "quantile",
        linestyles: List[str] = "solid",
        markers: Union[List[str], None] = ".",
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        title_fontsize: int = 10,
        legend_fontsize: int = 10,
        axes_label_fontsize: int = 10,
        axes_tick_fontsize: int = 10,
        label_identity_line: str = "_no_legend_",
        axis: Optional[plt.Axes] = None,
        dark_mode: bool = False
        ) -> Tuple[plt.Figure, plt.Axes]:
    """Plots (multiple) calibration curves for bins as obtained from sklearn calibration curve."""

    # Axis setup
    if axis is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#18181B" if dark_mode else "white")
    else:
        ax = axis
        fig = ax.figure

    if dark_mode:
        plt.style.use("dark_background")
        # Override the rcParams defaults that the style sets so that future figures
        # pick up the color we want (and savefig uses it by default).
        plt.rcParams["figure.facecolor"] = "#18181B"
        plt.rcParams["savefig.facecolor"] = "#18181B"
        plt.rcParams["axes.facecolor"] = "#18181B"

    # Identity line (perfect calibration): f(x)=x
    ax.plot([0, 1], [0, 1], color="white" if dark_mode else "black",
            linestyle="solid", label=label_identity_line)

    # Plot each calibration curve
    for i in range(len(predictions)):
        prob_true, prob_pred = calibration_curve(
            predictions[i], confidences[i], n_bins=n_bins, strategy=binning_strategy
        )
        ax.plot(
            prob_pred,
            prob_true,
            color=colors[i],
            marker=markers[i],
            linestyle=linestyles[i],
            label=labels[i]
        )

    # Styling: limit the axes to 0 and 1
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Average Confidence", labelpad=10, fontsize=axes_label_fontsize)
    ax.set_ylabel("Average Accuracy", labelpad=10, fontsize=axes_label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=axes_tick_fontsize)
    ax.set_title(title, pad=15, fontsize=title_fontsize)
    ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)

    # Styling: make the box visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("white" if dark_mode else "black")

    # Styling of the ticks
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_facecolor("#18181B" if dark_mode else "white")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax


def plot_accuracy_rejection_curve(
    title: str,
    predictions: List[List[Union[bool, int]]],
    confidences: List[List[float]],
    thresholds: List[float],
    linestyles: Union[List[str], str] = "solid",
    markers: Union[List[str], str] = "o",
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title_fontsize: int = 10,
    legend_fontsize: int = 10,
    axes_label_fontsize: int = 10,
    axes_tick_fontsize: int = 10,
    axis: Optional[plt.Axes] = None,
    return_df: bool = False
) -> Tuple[Figure | Figure, Axes, DataFrame] | Tuple[Figure | Any, Axes]:
    """
    Plots (multiple) accuracy–rejection curves for specified thresholds.
    """

    # Axis setup
    if axis is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        ax = axis
        fig = ax.figure

    n_estimators = len(predictions)

    # Normalize style arguments
    if isinstance(linestyles, str):
        linestyles = [linestyles] * n_estimators
    if isinstance(markers, str):
        markers = [markers] * n_estimators

    records = []

    for i in range(n_estimators):
        est_name = labels[i] if labels is not None else f"estimator_{i}"
        p = np.asarray(predictions[i], dtype=bool)
        c = np.asarray(confidences[i])

        rejected_fracs = []
        accuracies = []

        for t in thresholds:
            if t == 0:
                rejected_frac = 0.0
                frac_true = p.mean()
            elif t == 1:
                rejected_frac = 1.0
                frac_true = 1.0
            else:
                mask = c >= t
                rejected_frac = 1.0 - mask.mean()
                frac_true = p[mask].mean()

            rejected_fracs.append(rejected_frac)
            accuracies.append(frac_true)

            # Store for DataFrame
            records.append({
                "estimator": est_name,
                "threshold": t,
                "rejected_frac": rejected_frac,
                "accuracy": frac_true
            })

        ax.plot(
            rejected_fracs,
            accuracies,
            marker=markers[i],
            linestyle=linestyles[i],
            color=None if colors is None else colors[i],
            label=None if labels is None else labels[i],
        )

        # Threshold annotations
        for rf, acc, t in zip(rejected_fracs, accuracies, thresholds):
            ax.annotate(
                f"{t:.1f}",
                (rf, acc),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=axes_tick_fontsize + 2,
                color=None if colors is None else colors[i],
            )

    # Axes styling
    ax.set_facecolor("white")
    ax.set_xlim(-0.03, 1.03)
    ax.tick_params(left=True, labelleft=True)
    ax.set_xlabel("Rejection Rate", fontsize=axes_label_fontsize)
    ax.set_ylabel("Accuracy", fontsize=axes_label_fontsize)
    ax.set_xticks([j / 10 for j in range(11)])
    ax.set_yticks([j / 10 for j in range(5, 11)])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.2f}")
    )
    ax.tick_params(axis="both", which="major", labelsize=axes_tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize, pad=12)

    if labels is not None:
        ax.legend(frameon=False, fontsize=legend_fontsize)

    # Box styling
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    ax.grid(False)
    plt.tight_layout()

    if return_df:
        df = pd.DataFrame.from_records(records)
        return fig, ax, df
    else:
        return fig, ax
