import json

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.utils import aggregate, reduced_scores


def base_confidence(instances_df: pd.DataFrame) -> pd.Series:
    """
    Function to obtain the Base-Confidence.
    Any template can be used for this estimate.
    ---------------
    :param instances_df: Dataframe with the instance-level results per template.
    :returns: A pandas series of the Base-Confidence.
    """
    return instances_df["conf_scores"].apply(lambda x: np.max(x))


def margin_confidence(instances_df: pd.DataFrame) -> pd.Series:
    """
    Function to obtain the Confidence-Margin.
    Any template can be used for this estimate.
    ---------------
    :param instances_df: The dataframe with the instance-level results per template.
    :returns: A pandas series of the confidence margin.
    """
    return instances_df["conf_scores"].apply(lambda x: x[0] - x[1])


def average_confidence(
    instances_df: pd.DataFrame,
    strategies: Tuple[str] = ("voting", "min_conf", "max_conf"),
    votes_range: Tuple[int] = (2, 3, 4, 5)
) -> pd.DataFrame:
    """
    Computes the Average-Confidence per instance for specified aggregation strategies.
    If strategy equals 'voting', compute it for each votes_to_win in specified votes_range.
    ---------------
    :param instances_df: pd.DataFrame, instance-level results.
    :param strategies: Tuple[str], aggregation strategies for Average-Confidence.
    :param votes_range: Tuple[int], number of votes to win the voting aggregation.
    :returns: pd.DataFrame, with the Average-Confidence estimates added.
    """

    records = []
    # remove the epistemic templates
    grouped = instances_df[
        ~instances_df["template"].isin([5, 6, 7, 8, 9, 10, 11])].groupby(["relation", "instance"])

    for _, group in grouped:
        record = {
            "relation": group["relation"].iloc[0],
            "instance": group["instance"].iloc[0],
            "bear_score": group["correctly_predicted"].mean()
        }

        ground_truth = group["answer_idx"].iloc[0]

        for strategy in strategies:
            if strategy == "voting":
                for votes_to_win in votes_range:
                    pred_idx, _, fail = aggregate(
                        group, strategy=strategy, votes_to_win=votes_to_win
                    )

                    if fail:
                        avg_conf = 0.0
                        correct = False
                    else:
                        avg_conf = (
                            group.loc[group["predicted_index"] == pred_idx, "base_conf"].sum() / 5
                        )
                        correct = (pred_idx == ground_truth)

                    record[f"average_conf_{strategy}_{votes_to_win}"] = avg_conf
                    record[f"{strategy}_{votes_to_win}_correct"] = correct
                    record[f"{strategy}_{votes_to_win}_fail"] = fail
            else:
                pred_idx, _, fail = aggregate(group, strategy=strategy, votes_to_win=2)
                if fail:
                    avg_conf = 0.0
                    correct = False
                else:
                    avg_conf = group.loc[group["predicted_index"] == pred_idx, "base_conf"].sum() / 5
                    correct = (pred_idx == ground_truth)

                record[f"average_conf_{strategy}"] = avg_conf
                record[f"{strategy}_correct"] = correct
                record[f"{strategy}_fail"] = fail

        records.append(record)

    return pd.DataFrame(records)


def consistency_confidence(
    instances_df: pd.DataFrame,
    strategies: Tuple[str] = ("voting", "min_conf", "max_conf"),
    votes_range: Tuple[int] = (2, 3, 4, 5)
) -> pd.DataFrame:
    """
    Computes the Consistency-Confidence per instance for specified aggregation strategies.
    If strategy equals 'voting', compute it for each votes_to_win in specified votes_range.
    """

    records = []
    # remove epistemic templates
    grouped = instances_df[
        ~instances_df["template"].isin([5, 6, 7, 8, 9, 10, 11])].groupby(["relation", "instance"])

    for _, group in grouped:
        record = {
            "relation": group["relation"].iloc[0],
            "instance": group["instance"].iloc[0],
        }

        for strategy in strategies:
            if strategy == "voting":
                for votes_to_win in votes_range:
                    _, vote_count, fail = aggregate(
                        group, strategy=strategy, votes_to_win=votes_to_win
                    )
                    consistency = 0.0 if fail else vote_count / 5
                    record[f"consistency_conf_{strategy}_{votes_to_win}"] = consistency
            else:
                _, vote_count, fail = aggregate(group, strategy=strategy)
                consistency = 0.0 if fail else vote_count / 5
                record[f"consistency_conf_{strategy}"] = consistency

        records.append(record)

    return pd.DataFrame(records)


def marker_confidence(instances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to obtain a models prediction and associated base-confidences
    from epistemic marker injected templates (template indices from 5 onwards).
    ---------------
    :param instances_df: pd.DataFrame, instance-level results per template.
    :returns: pd.Dataframe, instance-level results with the epistemic results.
    """
    # templates 6 and 7 in the metadata are the verbalized injected ones
    weakener_df = instances_df[instances_df["template"] == 5].copy()
    strengthener_df = instances_df[instances_df["template"] == 6].copy()

    weakener_df.loc[:, "weakener_conf"] = base_confidence(weakener_df)
    strengthener_df.loc[:, "strengthener_conf"] = base_confidence(strengthener_df)

    weakener_df.loc[:, "weakener_correctly_predicted"] = weakener_df.apply(
        lambda row: np.argmax(row["pll_scores"]) == row["answer_idx"],
        axis=1
    )
    strengthener_df.loc[:, "strengthener_correctly_predicted"] = strengthener_df.apply(
        lambda row: np.argmax(row["pll_scores"]) == row["answer_idx"],
        axis=1
    )
    weakener_df = weakener_df[
        ["relation", "instance", "weakener_conf", "weakener_correctly_predicted"]
    ]
    strengthener_df = strengthener_df[
        ["relation", "instance", "strengthener_conf", "strengthener_correctly_predicted"]
    ]
    marker_results = weakener_df.merge(strengthener_df, on=["relation", "instance"])
    cols_to_keep = [
        "relation", "instance", "weakener_conf", "strengthener_conf",
        "weakener_correctly_predicted", "strengthener_correctly_predicted"
    ]
    # templates with the numerical confidence expressions
    c = 0
    for t in [7, 8, 9, 10, 11]:
        d = instances_df[instances_df["template"] == t].copy()
        d.loc[:, f"num_conf_{c}"] = base_confidence(d)
        d.loc[:, f"num_conf_{c}_correctly_predicted"] = d.apply(
            lambda row: np.argmax(row["pll_scores"]) == row["answer_idx"],
            axis=1
        )
        d = d[["relation", "instance", f"num_conf_{c}", f"num_conf_{c}_correctly_predicted"]]
        marker_results = marker_results.merge(d, on=["relation", "instance"])
        cols_to_keep.append(f"num_conf_{c}")
        cols_to_keep.append(f"num_conf_{c}_correctly_predicted")
        c += 25

    return marker_results[cols_to_keep]


def get_confidence_estimates(
        path: str,
        model: str = "gpt2",
        reduction: str = "sum",
        only_answers: bool = False
) -> pd.DataFrame:
    """
    Function to obtain the instance-level results: confidence estimates,
    correctness-labels etc. used for our analysis later.
    ---------------
    :param path: str, path to the raw model score.
    :param model: str, model to evaluate.
    :param reduction: str, reduction method for the token log-likelihoods.
    :param only_answers: bool, flag if only the answer tokens should correspond to the sentence-level log-likelihood.
    :returns: pd.DataFrame, final instance-level results for model.
    """

    # Load the raw instance-level results
    path_to_model_scores = f"{path}/{model}/scores.json"
    results = pd.read_json(path_to_model_scores, orient="records", lines=True)

    # Sentence-level log-likelihood per instance and answer option
    results["pll_scores"] = results.apply(
        lambda row: reduced_scores(row, reduction=reduction, only_answers=only_answers),
        axis=1
    )

    # For each instance and template we obtain predicted index, correctness and normalized softmax scores
    results["predicted_index"] = results["pll_scores"].apply(np.argmax)
    results["correctly_predicted"] = results.apply(
        lambda row: row["answer_idx"] == row["predicted_index"], axis=1
    )
    results["conf_scores"] = results["pll_scores"].apply(
        lambda scores: np.sort(softmax(scores))[::-1]
    )

    # Confidence estimates which rely on a single template
    results["base_conf"] = base_confidence(results)
    results["margin_conf"] = margin_confidence(results)

    # Confidence estimates which rely on multiple templates
    average_df = average_confidence(results)
    consistency_df = consistency_confidence(results)

    # Confidence estimate which use the injected templates
    marker_df = marker_confidence(results)

    # Merge results: confidence scores and correctness per instance for the different approaches
    summary_df = (
        results
        .merge(average_df, on=["relation", "instance"])
        .merge(consistency_df, on=["relation", "instance"])
        .merge(marker_df, on=["relation", "instance"])
        .query("template == 0")     # keep results from first template for single template estimates
        .drop(columns=[
            "template", "tokens", "sub_indices",
            "obj_indices", "template_indices", "conf_scores"
        ])
        .assign(model=model)
    )

    # Load and merge domain metadata
    with open("../../data/relation_info.json") as f:
        domain_data = json.load(f)
    domain_df = pd.DataFrame.from_dict(domain_data, orient="index").reset_index()
    domain_df.columns = ["relation", "domains"]

    return summary_df.merge(domain_df, on="relation", how="left")
