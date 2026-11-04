
import pandas as pd
import numpy as np

from config import REPORT_DIR


def main():

    print("\nSPECTRIX HMM PREDICTION DIAGNOSTICS")
    print("=" * 55)

    file_path = REPORT_DIR / "hmm_sequential_predictions.csv"
    df = pd.read_csv(file_path)

    required = [
        "channel_id",
        "actual_busy",
        "predicted_busy",
        "predicted_busy_probability",
        "sensing_observation"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    df["persistence_prediction"] = (
        df["sensing_observation"].map({"IDLE": 0, "BUSY": 1})
    )

    if df["persistence_prediction"].isna().any():
        raise ValueError("Invalid sensing observation found.")

    df["persistence_prediction"] = (
        df["persistence_prediction"].astype(int)
    )

    results = []

    for channel_id, group in df.groupby("channel_id", sort=True):

        hmm = group["predicted_busy"].to_numpy(dtype=int)
        baseline = group["persistence_prediction"].to_numpy(dtype=int)

        probability = group[
            "predicted_busy_probability"
        ].to_numpy(dtype=float)

        actual = group["actual_busy"].to_numpy(dtype=int)

        same_predictions = np.sum(hmm == baseline)
        different_predictions = np.sum(hmm != baseline)

        results.append({
            "Channel": channel_id,
            "Total Predictions": len(group),
            "Same Decisions": same_predictions,
            "Different Decisions": different_predictions,
            "Same Decision (%)": (
                same_predictions / len(group) * 100
            ),
            "Average Busy Probability (%)": (
                np.mean(probability) * 100
            ),
            "Actual Busy (%)": (
                np.mean(actual) * 100
            ),
            "HMM Busy Predictions (%)": (
                np.mean(hmm) * 100
            ),
            "Baseline Busy Predictions (%)": (
                np.mean(baseline) * 100
            )
        })

    result_df = pd.DataFrame(results)

    print("\nCHANNEL-WISE DIAGNOSTICS\n")
    print(result_df.round(3).to_string(index=False))

    output_path = REPORT_DIR / "hmm_prediction_diagnostics.csv"
    result_df.to_csv(output_path, index=False)

    print("\nDiagnostic report saved:")
    print(output_path)

    print("\nOVERALL DECISION COMPARISON")

    total = len(df)
    same = np.sum(
        df["predicted_busy"] == df["persistence_prediction"]
    )

    print(f"Total predictions: {total}")
    print(f"Identical decisions: {same}")
    print(f"Different decisions: {total - same}")
    print(f"Identical decision percentage: {same / total * 100:.3f}%")

    print("\nDIAGNOSTICS COMPLETED!")


if __name__ == "__main__":
    main()
