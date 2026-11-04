
import pandas as pd
import numpy as np

from config import REPORT_DIR


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    accuracy = (tp + tn) / len(actual)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 Score (%)": f1 * 100,
        "False Alarm Rate (%)":
            fp / max(fp + tn, 1) * 100,
        "Miss Detection Rate (%)":
            fn / max(fn + tp, 1) * 100
    }


def main():

    print("\nSPECTRIX HMM VS PERSISTENCE BASELINE")

    file_path = (
        REPORT_DIR / "hmm_sequential_predictions.csv"
    )

    df = pd.read_csv(file_path)

    required_columns = [
        "actual_busy",
        "predicted_busy",
        "sensing_observation"
    ]

    if not all(column in df.columns for column in required_columns):

        raise ValueError(
            "Required columns are missing from the prediction report."
        )

    actual = df["actual_busy"].to_numpy(dtype=int)

    hmm_predictions = df[
        "predicted_busy"
    ].to_numpy(dtype=int)

    # Persistence baseline:
    # Use current cooperative sensing observation
    # as the prediction for the next slot.
    persistence_predictions = (
        df["sensing_observation"]
        .map({
            "IDLE": 0,
            "BUSY": 1
        })
        .to_numpy(dtype=int)
    )

    if np.isnan(persistence_predictions).any():

        raise ValueError(
            "Unknown sensing observations found."
        )

    hmm_metrics = calculate_metrics(
        actual,
        hmm_predictions
    )

    baseline_metrics = calculate_metrics(
        actual,
        persistence_predictions
    )

    comparison = pd.DataFrame({
        "Metric": list(hmm_metrics.keys()),
        "HMM": list(hmm_metrics.values()),
        "Persistence Baseline": list(
            baseline_metrics.values()
        )
    })

    comparison["HMM Improvement"] = (
        comparison["HMM"]
        - comparison["Persistence Baseline"]
    )

    print("\nCOMPARISON RESULTS\n")

    print(
        comparison.round(3).to_string(index=False)
    )

    output_path = (
        REPORT_DIR / "hmm_vs_persistence_comparison.csv"
    )

    comparison.to_csv(
        output_path,
        index=False
    )

    print("\nComparison saved to:")
    print(output_path)


if __name__ == "__main__":

    main()
