
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import REPORT_DIR, GRAPH_DIR


def get_time_period(hour):
    """Classify a timestamp into one of six time periods."""

    if 0 <= hour < 5:
        return "Midnight"
    elif 5 <= hour < 8:
        return "Early Morning"
    elif 8 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 22:
        return "Evening"
    else:
        return "Late Night"


def calculate_metrics(group):
    """Calculate binary classification performance metrics."""

    actual = group["actual_busy"].to_numpy(dtype=int)
    predicted = group["predicted_busy"].to_numpy(dtype=int)

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

    false_alarm_rate = fp / max(fp + tn, 1)
    miss_detection_rate = fn / max(fn + tp, 1)

    return {
        "Samples": len(actual),
        "Accuracy (%)": accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1 Score (%)": f1 * 100,
        "False Alarm Rate (%)": false_alarm_rate * 100,
        "Miss Detection Rate (%)": miss_detection_rate * 100,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }


def main():

    print("\nSPECTRIX HMM TIME-PERIOD AND CHANNEL ANALYSIS")
    print("=" * 55)

    input_path = REPORT_DIR / "hmm_sequential_predictions.csv"

    df = pd.read_csv(input_path)

    required_columns = [
        "timestamp",
        "channel_id",
        "actual_busy",
        "predicted_busy"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if not df["actual_busy"].isin([0, 1]).all():
        raise ValueError("actual_busy must contain only 0 and 1.")

    if not df["predicted_busy"].isin([0, 1]).all():
        raise ValueError("predicted_busy must contain only 0 and 1.")

    df["hour"] = df["timestamp"].dt.hour
    df["time_period"] = df["hour"].apply(get_time_period)

    period_order = [
        "Midnight",
        "Early Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Late Night"
    ]

    # -------------------------------------------------
    # 1. TIME-PERIOD PERFORMANCE
    # -------------------------------------------------

    period_results = []

    for period in period_order:

        group = df[df["time_period"] == period]

        if not group.empty:
            metrics = calculate_metrics(group)
            metrics["Time Period"] = period
            period_results.append(metrics)

    period_df = pd.DataFrame(period_results)

    period_df = period_df[
        ["Time Period"] + [
            column for column in period_df.columns
            if column != "Time Period"
        ]
    ]

    period_path = REPORT_DIR / "hmm_time_period_metrics.csv"
    period_df.to_csv(period_path, index=False)

    print("\n1. TIME-PERIOD PERFORMANCE\n")
    print(
        period_df[
            [
                "Time Period",
                "Accuracy (%)",
                "Precision (%)",
                "Recall (%)",
                "F1 Score (%)",
                "False Alarm Rate (%)",
                "Miss Detection Rate (%)"
            ]
        ].round(3).to_string(index=False)
    )

    # -------------------------------------------------
    # 2. CHANNEL-WISE PERFORMANCE
    # -------------------------------------------------

    channel_results = []

    for channel_id, group in df.groupby("channel_id", sort=True):

        metrics = calculate_metrics(group)
        metrics["Channel"] = channel_id
        channel_results.append(metrics)

    channel_df = pd.DataFrame(channel_results)

    channel_df = channel_df[
        ["Channel"] + [
            column for column in channel_df.columns
            if column != "Channel"
        ]
    ]

    channel_path = REPORT_DIR / "hmm_channel_detailed_metrics.csv"
    channel_df.to_csv(channel_path, index=False)

    print("\n2. CHANNEL-WISE PERFORMANCE\n")
    print(
        channel_df[
            [
                "Channel",
                "Accuracy (%)",
                "Precision (%)",
                "Recall (%)",
                "F1 Score (%)",
                "False Alarm Rate (%)",
                "Miss Detection Rate (%)"
            ]
        ].round(3).to_string(index=False)
    )

    # -------------------------------------------------
    # 3. CHANNEL VS TIME-PERIOD ACCURACY
    # -------------------------------------------------

    accuracy_matrix = df.pivot_table(
        index="channel_id",
        columns="time_period",
        values="actual_busy",
        aggfunc="count"
    )

    # Replace counts with accuracy values.
    accuracy_matrix = pd.DataFrame(
        index=sorted(df["channel_id"].unique()),
        columns=period_order,
        dtype=float
    )

    for channel_id in accuracy_matrix.index:

        for period in period_order:

            group = df[
                (df["channel_id"] == channel_id) &
                (df["time_period"] == period)
            ]

            if not group.empty:
                metrics = calculate_metrics(group)
                accuracy_matrix.loc[channel_id, period] = (
                    metrics["Accuracy (%)"]
                )

    accuracy_path = REPORT_DIR / "hmm_channel_period_accuracy.csv"
    accuracy_matrix.to_csv(accuracy_path)

    # -------------------------------------------------
    # 4. TIME-PERIOD ACCURACY GRAPH
    # -------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.bar(
        period_df["Time Period"],
        period_df["Accuracy (%)"]
    )

    plt.title("HMM Prediction Accuracy Across Time Periods")
    plt.xlabel("Time Period")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=20)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "hmm_time_period_accuracy.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # 5. CHANNEL-TIME-PERIOD HEATMAP
    # -------------------------------------------------

    plt.figure(figsize=(12, 6))

    image = plt.imshow(
        accuracy_matrix.to_numpy(dtype=float),
        cmap="YlGnBu",
        aspect="auto",
        vmin=0,
        vmax=100
    )

    plt.colorbar(image, label="Accuracy (%)")

    plt.xticks(
        range(len(period_order)),
        period_order,
        rotation=20
    )

    plt.yticks(
        range(len(accuracy_matrix.index)),
        accuracy_matrix.index
    )

    plt.title("HMM Accuracy: Channel vs Time Period")
    plt.xlabel("Time Period")
    plt.ylabel("Channel")

    for row in range(len(accuracy_matrix.index)):
        for col in range(len(period_order)):

            value = accuracy_matrix.iloc[row, col]

            if pd.notna(value):
                plt.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center"
                )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "hmm_channel_period_heatmap.png",
        dpi=300
    )

    plt.close()

    # -------------------------------------------------
    # COMPLETION
    # -------------------------------------------------

    print("\n3. GENERATED FILES")
    print("-" * 40)

    print("Reports:")
    print(period_path)
    print(channel_path)
    print(accuracy_path)

    print("\nGraphs:")
    print(GRAPH_DIR / "hmm_time_period_accuracy.png")
    print(GRAPH_DIR / "hmm_channel_period_heatmap.png")

    print("\nHMM TIME-PERIOD ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
