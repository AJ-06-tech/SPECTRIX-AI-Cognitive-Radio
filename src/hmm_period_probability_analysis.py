
import pandas as pd
import numpy as np

from config import REPORT_DIR


def get_time_period(hour):

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


def calculate_brier(actual, probability):

    return np.mean((actual - probability) ** 2)


def calculate_log_loss(actual, probability):

    probability = np.clip(probability, 1e-10, 1 - 1e-10)

    return -np.mean(
        actual * np.log(probability)
        + (1 - actual) * np.log(1 - probability)
    )


def main():

    print("\nSPECTRIX HMM TIME-PERIOD PROBABILITY ANALYSIS")
    print("=" * 60)

    file_path = REPORT_DIR / "hmm_sequential_predictions.csv"

    df = pd.read_csv(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["time_period"] = df["timestamp"].dt.hour.apply(
        get_time_period
    )

    period_order = [
        "Midnight",
        "Early Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Late Night"
    ]

    results = []

    for period in period_order:

        group = df[df["time_period"] == period]

        actual = group["actual_busy"].to_numpy(dtype=int)

        probability = group[
            "predicted_busy_probability"
        ].to_numpy(dtype=float)

        if len(group) == 0:
            continue

        results.append({

            "Time Period": period,

            "Samples": len(group),

            "Brier Score": calculate_brier(
                actual,
                probability
            ),

            "Log Loss": calculate_log_loss(
                actual,
                probability
            )
        })

    result_df = pd.DataFrame(results)

    print("\nTIME-PERIOD PROBABILITY PERFORMANCE\n")

    print(result_df.round(5).to_string(index=False))

    output_path = REPORT_DIR / "hmm_period_probability_metrics.csv"

    result_df.to_csv(output_path, index=False)

    print("\nReport saved:")
    print(output_path)

    print("\nTIME-PERIOD PROBABILITY ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
