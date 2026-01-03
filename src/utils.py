import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union

from lm_pub_quiz import Evaluator, Dataset
from scipy.special import softmax

from src.metrics import adaptive_calibration_error, weighted_average


def reduced_scores(
        row,
        reduction: str = "sum",
        only_answers: bool = False
        ) -> List[float]:
    """
    Helper function to obtain a sentence-level log-likelihood based on different
    reduction strategies (Sum, Sum (A), Mean, Mean (A)). Scores are obtained using
    LM-PUB-QUIZ.
    ---------------
    :param row: Corresponds to the row of a single instance.
    :param reduction: The reduction strategy used for token log-likelihood reduction.
    :param only_answers: Flag if only the answer tokens should correspond to the sentence-level log-likelihood.
    :returns: List of sentence-level log-likelihoods per instance, one for each answer option.
    """
    # Get scores and indices
    scores = row["pll_scores"]
    sub = row["sub_indices"]
    obj = row["obj_indices"]
    template = row["template_indices"]

    # Combining indices per answer option
    combined = obj if only_answers else [sub[i] + obj[i] + template[i] for i in range(len(scores))]

    sentence_level_log_likelihoods = []
    for i in range(len(scores)):
        scores_to_reduce = [scores[i][j] for j in combined[i]]
        sentence_level_log_likelihood = sum(scores_to_reduce) if reduction == "sum" else np.mean(scores_to_reduce)
        sentence_level_log_likelihoods.append(sentence_level_log_likelihood)

    return sentence_level_log_likelihoods


def aggregate(
    group: pd.DataFrame,
    strategy: str,
    votes_to_win: int = 5
) -> Tuple[Optional[int], int, bool]:
    """
    Compute the final answer for a group of template predictions.

    Returns:
        - predicted_index: the selected answer (None if failed)
        - vote_count: number of templates agreeing with selected answer
        - fail: True if aggregation failed
    """

    if strategy == "voting":
        if not 2 <= votes_to_win <= 5:
            raise ValueError("votes_to_win must be between 2 and 5")

        counts = group["predicted_index"].value_counts()
        max_votes = counts.max()

        # Fail if threshold not met
        if max_votes < votes_to_win:
            return None, max_votes, True

        # Check uniqueness of the maximum
        top = counts[counts == max_votes]
        if len(top) != 1:
            return None, max_votes, True

        pred_idx = top.index[0]
        return pred_idx, max_votes, False

    elif strategy == "min_conf":
        row = group.loc[group["base_conf"].idxmin()]
        pred_idx = row["predicted_index"]
        vote_count = (group["predicted_index"] == pred_idx).sum()
        return pred_idx, vote_count, False

    elif strategy == "max_conf":
        row = group.loc[group["base_conf"].idxmax()]
        pred_idx = row["predicted_index"]
        vote_count = (group["predicted_index"] == pred_idx).sum()
        return pred_idx, vote_count, False

    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")


def plot_accuracy_vs_metric(results_df,
                            estimator: str = "base_conf",
                            metric: str = "brier_score",
                            title: str = "",
                            axis: Optional[plt.Axes] = None
                            ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Helper function to plot accuracy vs calibration metric (ACE, brier score)
    per model family for a given estimator.
    """
    results_df = results_df[results_df["estimator"] == estimator]

    if axis is None:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    else:
        ax = axis
        fig = ax.figure

    # Plot the accuracies against the metrics per family
    for family, group in results_df.groupby("family"):
        group.sort_values("accuracy", inplace=True)
        ax.plot(
            group["accuracy"],
            group[metric],
            marker=".",
            markersize=20,
            linestyle="dotted",
            linewidth=8,
            label=family
        )

    ax.set_xlabel("Accuracy", labelpad=10, fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), labelpad=10, fontsize=12)
    ax.set_title(title, pad=15, fontsize=14)
    ax.legend(title="", loc="upper right", frameon=False, fontsize=13)

    # Make the box around the plot visible
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("black")

    # More styling: ticks point outwards, white background, no gridlines
    ax.tick_params(direction="out", length=4, width=1)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.set_facecolor("white")
    ax.grid(False)

    plt.tight_layout()
    return fig, ax


def evaluate_reductions(models: Union[str, List[str]] = "gpt2") -> pd.DataFrame:
    """
    Helper function for our first experiment evaluating which reduction method is most
    effective. For each model and reduction strategy finds the ACE and accuracy. Only the
    first template and base-confidence is used.
    """
    if isinstance(models, str):
        models = [models]

    reductions = [
        "pll_scores_sum",
        "pll_scores_avg",
        "pll_scores_answer_sum",
        "pll_scores_answer_avg",
    ]

    all_summary_rows = []

    for model in models:
        output_path = f"../scores/{model}"
        file_path = f"{output_path}/scores.json"
        results = pd.read_json(file_path, orient="records", lines=True)
        model_type = results["model_type"].iloc[0]
        results = results[results["template"] == 0]     # restrict analysis to the first template (base_conf)

        results["pll_scores_sum"] = results.apply(
            lambda row: reduced_scores(row, reduction="sum", only_answers=False),
            axis=1)
        results["pll_scores_avg"] = results.apply(
            lambda row: reduced_scores(row, reduction="mean", only_answers=False),
            axis=1)
        results["pll_scores_answer_sum"] = results.apply(
            lambda row: reduced_scores(row, reduction="sum", only_answers=True),
            axis=1)
        results["pll_scores_answer_avg"] = results.apply(
            lambda row: reduced_scores(row, reduction="mean", only_answers=True),
            axis=1)

        for reduction in reductions:
            results[f"{reduction}_correctly_predicted"] = results.apply(
                lambda row: row["answer_idx"] == np.argmax(row[reduction]),
                axis=1
            )
            results[f"{reduction}_confs"] = results.apply(
                lambda row: np.sort(softmax(row[reduction]))[::-1],
                axis=1
            )
            results[f"{reduction}_base_conf"] = results.apply(
                lambda row: row[f"{reduction}_confs"][0],
                axis=1
            )

            # Accuracy as the weighted average over all relations
            grouped = results.groupby("relation").agg({
                "instance": "count",
                f"{reduction}_correctly_predicted": "mean"
            }).reset_index()
            accuracy = weighted_average(grouped, f"{reduction}_correctly_predicted")
            predictions = results[f"{reduction}_correctly_predicted"].tolist()
            confidences = results[f"{reduction}_base_conf"].tolist()
            ace = round(adaptive_calibration_error(predictions, confidences), 6)

            all_summary_rows.append({
                "model": model,
                "model_type": model_type,
                "reduction": reduction,
                "accuracy": accuracy,
                "ace": ace
            })

    return pd.DataFrame(all_summary_rows)


def get_model_scores(
        model_id: str = "openai-community/gpt2",
        model_type: str = "CLM",
        path_to_data: str = "../data/BEAR",
        device: str = "cuda",
        templates: Union[int, List[int]] = 0,
        batch_size: int = 32
        ) -> None:
    """
    Function used to obtain the model score using LM-PUB-QUIZ for specified templates.
    Stores the raw (unreduced) scores per instance and template.
    """
    output_path = f"../scores/BEAR/{model_id.split('/')[-1]}"
    file_path = f"{output_path}/scores.json"

    evaluator = Evaluator.from_model(model=model_id, model_type=model_type, device=device)
    bear = Dataset.from_path(path_to_data)
    results = pd.DataFrame()

    if isinstance(templates, int):
        templates = [templates]
    for template in templates:
        temp_df = (
            evaluator
            .evaluate_dataset(bear, template_index=template, batch_size=batch_size, reduction=None)
            .joined_instance_table()
            .reset_index()
            .assign(template=template)
        )
        results = pd.concat([results, temp_df], ignore_index=True)

    results["model_type"] = model_type
    results["model"] = model_id.split("/")[-1]
    os.makedirs(output_path, exist_ok=True)
    results.to_json(file_path, orient="records", lines=True)
    print(f"Scores saved to: {file_path}")


def subsample_answer_options(total_len: int, answer_index: int, sample_size: int):
    """
    Function to draw a sample of size 'sample_size' of the answer options of an instance.
    """
    if sample_size > total_len:
        raise ValueError("Sample size > total available scores.")

    all_indices = np.arange(total_len)
    remaining = np.delete(all_indices, answer_index)

    # Randomly choose sample_size - 1 from remaining
    choice = np.random.choice(remaining, size=sample_size - 1, replace=False)
    subsample = np.concatenate(([answer_index], choice))
    return subsample


def get_factor_results(
        list_of_models: List[str],
        reduction_strategy: str = "sum"
) -> pd.DataFrame:
    """
    Loads the results per model obtained from Wiki-FACTOR (https://github.com/AI21Labs/factor) and
    adds the estimates of Base- and Margin-Confidence.
    :param list_of_models: List[str], the models to evaluate.
    :param reduction_strategy: str, reduction strategy applied to the token log-likelihoods.
    :return: pd.DataFrame, per-model and per-question results.
    """
    results = []
    for m in list_of_models:
        r = pd.read_json(
            f"../../scores/FACTOR/{reduction_strategy}/{m}",
            orient="records",
            lines=True
        )
        r["model"] = m.removesuffix(".jsonl")
        r["softmax_scores"] = r["scores"].apply(lambda s: softmax(-1 * np.array(s)))
        r["predicted_answer"] = r["softmax_scores"].apply(np.argmax)
        r["correctly_predicted"] = r["predicted_answer"] == 0
        r["base_conf"] = r.apply(lambda x: np.max(x["softmax_scores"]), axis=1)
        r["margin_conf"] = r["softmax_scores"].apply(lambda s: np.max(s) - np.partition(s, -2)[-2])

        results.append(r)
    return pd.concat(results, ignore_index=True)
