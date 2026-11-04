
import numpy as np
import pandas as pd

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from hmm_predictor import HMMPredictor

from config import REPORT_DIR


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    false_alarm_rate = fp / max(fp + tn, 1)
    miss_detection_rate = fn / max(fn + tp, 1)

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100,
        "false_alarm_rate": false_alarm_rate * 100,
        "miss_detection_rate": miss_detection_rate * 100,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }


def main():

    print("\n" + "=" * 75)
    print(" SPECTRIX HMM SEQUENTIAL EVALUATION")
    print("=" * 75)

    # --------------------------------------------------
    # INITIALIZE MODULES
    # --------------------------------------------------

    print("\nInitializing evaluation modules...")

    channel_simulator = ChannelSimulator()
    sensing_engine = SpectrumSensing()
    hmm_predictor = HMMPredictor()

    # --------------------------------------------------
    # GET HELD-OUT TEST TIMESTAMPS
    # --------------------------------------------------

    all_timestamps = pd.DatetimeIndex(
        channel_simulator.get_all_timestamps()
    )

    test_timestamps = all_timestamps[
        all_timestamps >= hmm_predictor.testing_start
    ]

    if len(test_timestamps) < 2:
        raise ValueError("Insufficient test timestamps.")

    print("\nTesting timestamps:", len(test_timestamps))
    print("Testing starts:", test_timestamps[0])
    print("Testing ends:", test_timestamps[-1])

    print(
        "\nPredictions will be evaluated against the "
        "following 10-second slot."
    )

    # --------------------------------------------------
    # SEQUENTIAL EVALUATION
    # --------------------------------------------------

    actual_values = {
        channel_id: []
        for channel_id in hmm_predictor.transition_matrices
    }

    predicted_values = {
        channel_id: []
        for channel_id in hmm_predictor.transition_matrices
    }

    prediction_records = []

    total_steps = len(test_timestamps) - 1

    print("\nStarting sequential HMM evaluation...")

    for index in range(total_steps):

        current_time = test_timestamps[index]
        next_time = test_timestamps[index + 1]

        current_time_string = current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        next_time_string = next_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # STEP 1: Obtain actual current channel status
        current_status = channel_simulator.get_channel_status(
            current_time_string
        )

        # STEP 2: Perform individual sensing
        sensing_reports = sensing_engine.perform_sensing(
            current_status
        )

        # STEP 3: Cooperative sensing
        cooperative_results = sensing_engine.cooperative_sensing(
            sensing_reports
        )

        sensing_decisions = {
            channel_id: result["decision"]
            for channel_id, result in cooperative_results.items()
        }

        # STEP 4: Update HMM and predict next slot
        predictions = hmm_predictor.process_observations(
            sensing_decisions
        )

        # STEP 5: Obtain actual occupancy of the NEXT slot
        next_status = channel_simulator.get_channel_status(
            next_time_string
        )

        # STEP 6: Compare predicted and actual states
        for channel_id, result in predictions.items():

            predicted_probability = result["probability_busy"]

            predicted_state = int(
                predicted_probability >= 0.5
            )

            actual_state = int(
                next_status[channel_id]["status"] != "AVAILABLE"
            )

            actual_values[channel_id].append(actual_state)

            predicted_values[channel_id].append(
                predicted_state
            )

            prediction_records.append({
                "timestamp": next_time_string,
                "channel_id": channel_id,
                "actual_busy": actual_state,
                "predicted_busy": predicted_state,
                "predicted_busy_probability": predicted_probability,
                "sensing_observation": sensing_decisions[channel_id]
            })

        # Progress display
        if (index + 1) % 2000 == 0:

            print(
                f"Processed {index + 1}/{total_steps} slots"
            )

    print("\nSequential evaluation completed!")

    # --------------------------------------------------
    # CALCULATE PER-CHANNEL METRICS
    # --------------------------------------------------

    channel_results = []

    for channel_id in actual_values:

        metrics = calculate_metrics(
            actual_values[channel_id],
            predicted_values[channel_id]
        )

        metrics["channel_id"] = channel_id

        channel_results.append(metrics)

    channel_metrics_df = pd.DataFrame(channel_results)

    # --------------------------------------------------
    # CALCULATE OVERALL METRICS
    # --------------------------------------------------

    all_actual = np.concatenate([
        np.asarray(values)
        for values in actual_values.values()
    ])

    all_predicted = np.concatenate([
        np.asarray(values)
        for values in predicted_values.values()
    ])

    overall_metrics = calculate_metrics(
        all_actual,
        all_predicted
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 75)
    print(" HMM OVERALL PERFORMANCE")
    print("=" * 75)

    print("Total predictions:", len(all_actual))

    for metric, value in overall_metrics.items():

        if isinstance(value, float):

            print(
                metric.replace("_", " ").title(),
                ":",
                round(value, 3),
                "%"
            )

        else:

            print(
                metric.replace("_", " ").title(),
                ":",
                value
            )

    print("\n" + "=" * 75)
    print(" CHANNEL-WISE HMM PERFORMANCE")
    print("=" * 75)

    print(
        channel_metrics_df[
            [
                "channel_id",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "false_alarm_rate",
                "miss_detection_rate"
            ]
        ].round(3).to_string(index=False)
    )

    # --------------------------------------------------
    # SAVE REPORTS
    # --------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_df = pd.DataFrame(prediction_records)

    prediction_path = (
        REPORT_DIR / "hmm_sequential_predictions.csv"
    )

    channel_metrics_path = (
        REPORT_DIR / "hmm_channel_metrics.csv"
    )

    overall_metrics_path = (
        REPORT_DIR / "hmm_overall_metrics.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    channel_metrics_df.to_csv(
        channel_metrics_path,
        index=False
    )

    pd.DataFrame([
        overall_metrics
    ]).to_csv(
        overall_metrics_path,
        index=False
    )

    print("\nEvaluation reports saved:")

    print(prediction_path)
    print(channel_metrics_path)
    print(overall_metrics_path)

    print("\nHMM SEQUENTIAL EVALUATION COMPLETED")


if __name__ == "__main__":

    main()
