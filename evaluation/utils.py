import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve


def master_dataframe(dict_of_metrics: dict):
    df = pd.DataFrame.from_dict(
        {
            (i, j, k): dict_of_metrics[i][j][k]
            for i in dict_of_metrics.keys()
            for j in dict_of_metrics[i].keys()
            for k in dict_of_metrics[i][j].keys()
        },
        orient="index",
    )
    df.index.names = ["dataset", "scoring_method", "model"]
    df = df.reset_index()
    return df


def get_per_metric_dataframe(df: pd.DataFrame, list_of_metrics: list, delta=False):
    assert set(list_of_metrics).issubset(
        list(df.columns)
    ), f"some elements in list_of_metrics {list_of_metrics} are not avaialble in column of dataframe {df.columns}"

    change = "Δ" if delta else ""

    dataframe_dict = {}
    for metric in list_of_metrics:
        metric_df = (
            pd.pivot_table(
                df,
                values=f"{change+metric}",
                index=["dataset", "model"],
                columns=["scoring_method"],
            )
            .reset_index()
            .sort_values(by=["dataset", "model"])
        )
        dataframe_dict[metric] = metric_df
    return dataframe_dict


def create_metricwise_dict(df, list_of_metrics, delta=False):

    metrics_considered = ["auroc"] if list_of_metrics is None else list_of_metrics
    df["dataset_model"] = df.dataset + "| " + df.model
    metrics_dict = get_per_metric_dataframe(
        df, delta=delta, list_of_metrics=metrics_considered
    )
    return metrics_dict


def draw_dot_plot(
    df: pd.DataFrame,
    scores_considered: list,
    *,
    title: str = "AUROC",
    figsize: tuple = (8, 6),
    alpha=1,
    markersize=8,
):
    plt.rcParams["figure.figsize"] = figsize
    df = df.sort_values(by=["dataset"])  # sort on y-axis
    df["dataset_model"] = df.dataset + " | " + df.model  # title fot y axis per model
    labels = df["dataset_model"].tolist()
    x = np.arange(len(labels))  # the label locations

    score = []
    for score_type in scores_considered:
        score.append(df[score_type])

    jf = 0.15  # jitter factor

    fig, ax = plt.subplots()
    for i in range(len(score)):
        _ = ax.plot(
            score[i],
            x + np.random.uniform(-jf, jf),
            marker="o",
            linestyle="None",
            label=scores_considered[i],
            markersize=markersize,
            alpha=alpha,
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Dataset | Model")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_yticks(x, labels)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.legend(fontsize=10)

    # plot horizontal lines between each model group
    a = [(x[i] + x[i + 1]) / 2 for i in range(4, len(x) - 4, 5)]
    [ax.axhline(y=i, linestyle="solid", c="black", linewidth=0.75) for i in a]

    fig.tight_layout()
    plt.show()


def lift_at_k(y_true: np.array, y_score: np.array, k: int = 100) -> float:
    """Compute Lift at K evaluation metric"""

    sort_indices = np.argsort(y_score)
    # compute lift for the top k values
    lift_at_k = y_true[sort_indices][-k:].mean() / y_true.mean()
    return lift_at_k


def compute_metrics(
    scores: np.array, true_error: np.array, true_diff: np.array
) -> dict:
    """
    Uses passed in `scores` of length (N,) and `true_error` mask of length (N,) to calculate metrics for the data.

    Parameters
    ----------
    scores : np.ndarray
    Scores per example with values [0,1]. Smaller values indicating examples that are more likeley to have error.
    Can come directly from calclation like `scores = get_label_quality_scores()`.

    true_error : np.ndarray
    Binary mask indicating which examples have error with 1 = error and 0 = no error.

    metrics_metadata : dict
    Additional information to append to the calculated metrics dict.

    Returns
    -------
    metrics : dict
    A dictionary of computed metrics given `scores` and `true_error`.
    """
    # compute precision-recall curve using quality scores
    precision, recall, _ = precision_recall_curve(true_error, 1 - scores)
    fpr, tpr, _ = roc_curve(true_error, 1 - scores)

    # compute prc auc scores
    auprc = auc(recall, precision)

    # compute accuracy of detecting errors
    auroc = roc_auc_score(true_error, 1 - scores)

    # lift at K where K = number of errors
    lift_at_num_errors = lift_at_k(true_error, 1 - scores, k=true_error.sum())

    # lift at k=100
    lift_at_100 = lift_at_k(true_error, 1 - scores, k=100)

    # spearman correration between scores and true_diff
    spearman_corr = spearmanr(1 - scores, abs(true_diff))[0]

    metrics = {
        "dataset_num_samples": len(scores),
        "dataset_num_errors": true_error.sum(),
        "auroc": auroc,
        "auprc": auprc,
        "lift_at_num_errors": lift_at_num_errors,
        "lift_at_100": lift_at_100,
        "spearman_corr": spearman_corr,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
    }

    return metrics