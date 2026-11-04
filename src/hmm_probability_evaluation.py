
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import REPORT_DIR, GRAPH_DIR


def calculate_brier_score(actual, probability):
    return np.mean((probability - actual) ** 2)


def calculate_log_loss(actual, probability):

    epsilon = 1e-10

    probability = np.clip(
        probability,
        epsilon,
        1 - epsilon
    )

    return -np.mean(
        actual * np.log(probability)
        + (1 - actual) * np.log(1 - probability)
    )


def get_persistence_probability(observation):

    # Theoretical cooperative sensing probabilities.
    # These assume independent sensor errors.

    pd_coop = 0.99144
    pfa_coop = 0.00116

    if observation == "BUSY":
        return pd_coop

    return pfa_coop


def main():

    print("\nSPECTRIX HMM PROBABILITY EVALUATION")
    print("=" * 55)

    input_path = REPORT_DIR / "hmm_sequential_predictions.csv"

    df = pd.read_csv(input_path)

    required = [
        "actual_busy",
        "predicted_busy_probability",
        "sensing_observation",
        "channel_id"
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    actual = df["actual_busy"].to_numpy(dtype=int)

    hmm_probability = df[
        "predicted_busy_probability"
    ].to_numpy(dtype=float)

    if not np.isfinite(hmm_probability).all():
        raise ValueError("Invalid HMM probability values.")

    if ((hmm_probability < 0) | (hmm_probability > 1)).any():
        raise ValueError("HMM probabilities must be between 0 and 1.")

    persistence_probability = np.array([
        get_persistence_probability(value)
        for value in df["sensing_observation"]
    ])

    # -------------------------------------------------
    # 1. OVERALL PROBABILITY METRICS
    # -------------------------------------------------

    results = []

    for model_name, probability in [
        ("HMM", hmm_probability),
        ("Cooperative Sensing Probability Baseline",
         persistence_probability)
    ]:

        results.append({
            "Model": model_name,
            "Brier Score": calculate_brier_score(
                actual, probability
            ),
            "Log Loss": calculate_log_loss(
                actual, probability
            )
        })

    metrics_df = pd.DataFrame(results)

    print("\n1. OVERALL PROBABILITY PERFORMANCE\n")
    print(metrics_df.round(5).to_string(index=False))

    metrics_path = REPORT_DIR / "hmm_probability_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)

    # -------------------------------------------------
    # 2. CALIBRATION DATA
    # -------------------------------------------------

    df["probability_bin"] = pd.cut(
        df["predicted_busy_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True
    )

    calibration_df = (
        df.groupby("probability_bin", observed=False)
        .agg(
            prediction_count=("actual_busy", "count"),
            mean_predicted_probability=(
                "predicted_busy_probability", "mean"
            ),
            actual_busy_frequency=("actual_busy", "mean")
        )
        .reset_index()
    )

    calibration_path = REPORT_DIR / "hmm_calibration_data.csv"

    calibration_df.to_csv(calibration_path, index=False)

    print("\n2. CALIBRATION TABLE\n")
    print(calibration_df.to_string(index=False))

    # -------------------------------------------------
    # 3. CALIBRATION GRAPH
    # -------------------------------------------------

    valid = calibration_df[
        calibration_df["prediction_count"] > 0
    ]

    plt.figure(figsize=(8, 7))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.plot(
        valid["mean_predicted_probability"],
        valid["actual_busy_frequency"],
        marker="o",
        label="HMM calibration"
    )

    plt.xlabel("Mean Predicted Busy Probability")
    plt.ylabel("Actual Busy Frequency")

    plt.title("HMM Probability Calibration")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    graph_path = GRAPH_DIR / "hmm_probability_calibration.png"

    plt.savefig(graph_path, dpi=300)
    plt.close()

    # -------------------------------------------------
    # 4. CHANNEL-WISE BRIER SCORES
    # -------------------------------------------------

    channel_results = []

    for channel_id, group in df.groupby("channel_id", sort=True):

        channel_actual = group["actual_busy"].to_numpy(dtype=int)

        channel_probability = group[
            "predicted_busy_probability"
        ].to_numpy(dtype=float)

        channel_results.append({
            "Channel": channel_id,
            "Brier Score": calculate_brier_score(
                channel_actual,
                channel_probability
            ),
            "Log Loss": calculate_log_loss(
                channel_actual,
                channel_probability
            )
        })

    channel_df = pd.DataFrame(channel_results)

    channel_path = REPORT_DIR / "hmm_channel_probability_metrics.csv"

    channel_df.to_csv(channel_path, index=False)

    print("\n3. CHANNEL-WISE PROBABILITY PERFORMANCE\n")
    print(channel_df.round(5).to_string(index=False))

    print("\nGENERATED FILES")
    print(metrics_path)
    print(calibration_path)
    print(channel_path)
    print(graph_path)

    print("\nHMM PROBABILITY EVALUATION COMPLETED!")


if __name__ == "__main__":
    main()
