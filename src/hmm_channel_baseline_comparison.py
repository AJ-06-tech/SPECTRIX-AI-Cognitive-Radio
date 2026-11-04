
import pandas as pd
import numpy as np

from config import REPORT_DIR


def calculate_metrics(actual, predicted):

    tp = np.sum((actual == 1) & (predicted == 1))
    tn = np.sum((actual == 0) & (predicted == 0))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    accuracy = (tp + tn) / max(len(actual), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "Accuracy (%)": accuracy * 100,
        "F1 Score (%)": f1 * 100,
        "False Alarm Rate (%)": fp / max(fp + tn, 1) * 100,
        "Miss Detection Rate (%)": fn / max(fn + tp, 1) * 100
    }


def main():

    print("\nHMM VS PERSISTENCE: CHANNEL-WISE COMPARISON")
    print("=" * 55)

    file_path = REPORT_DIR / "hmm_sequential_predictions.csv"
    df = pd.read_csv(file_path)

    required = [
        "channel_id",
        "actual_busy",
        "predicted_busy",
        "sensing_observation"
    ]

    if not all(column in df.columns for column in required):
        raise ValueError("Required columns are missing.")

    actual = df["actual_busy"].to_numpy(dtype=int)
    hmm = df["predicted_busy"].to_numpy(dtype=int)

    persistence = df["sensing_observation"].map({
        "IDLE": 0,
        "BUSY": 1
    })

    if persistence.isna().any():
        raise ValueError("Unknown sensing observation found.")

    df["persistence_prediction"] = persistence.astype(int)

    results = []

    for channel_id, group in df.groupby("channel_id", sort=True):

        actual = group["actual_busy"].to_numpy(dtype=int)
        hmm = group["predicted_busy"].to_numpy(dtype=int)
        persistence = group["persistence_prediction"].to_numpy(dtype=int)

        hmm_metrics = calculate_metrics(actual, hmm)
        baseline_metrics = calculate_metrics(actual, persistence)

        results.append({
            "Channel": channel_id,
            "HMM Accuracy (%)": hmm_metrics["Accuracy (%)"],
            "Baseline Accuracy (%)": baseline_metrics["Accuracy (%)"],
            "Accuracy Improvement": (
                hmm_metrics["Accuracy (%)"] -
                baseline_metrics["Accuracy (%)"]
            ),
            "HMM F1 (%)": hmm_metrics["F1 Score (%)"],
            "Baseline F1 (%)": baseline_metrics["F1 Score (%)"],
            "F1 Improvement": (
                hmm_metrics["F1 Score (%)"] -
                baseline_metrics["F1 Score (%)"]
            ),
            "HMM False Alarm (%)": hmm_metrics["False Alarm Rate (%)"],
            "Baseline False Alarm (%)": baseline_metrics["False Alarm Rate (%)"],
            "HMM Miss Detection (%)": hmm_metrics["Miss Detection Rate (%)"],
            "Baseline Miss Detection (%)": baseline_metrics["Miss Detection Rate (%)"],
            "Miss Detection Reduction": (
                baseline_metrics["Miss Detection Rate (%)"] -
                hmm_metrics["Miss Detection Rate (%)"]
            )
        })

    result_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "hmm_channel_baseline_comparison.csv"
    result_df.to_csv(output_path, index=False)

    print("\nCHANNEL-WISE COMPARISON\n")

    print(result_df.round(3).to_string(index=False))

    print("\nReport saved to:")
    print(output_path)

    print("\nANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
