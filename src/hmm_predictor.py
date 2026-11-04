
import numpy as np
import pandas as pd

from config import (
    RAW_DATA_DIR,
    CHANNEL_IDS,
    RANDOM_SEED,
    COOPERATIVE_DETECTION_PROBABILITY,
    COOPERATIVE_FALSE_ALARM_PROBABILITY
)


class HMMPredictor:

    def __init__(self):

        print("Initializing Observation-Aware HMM...")

        self.dataset_path = RAW_DATA_DIR / "occupancy_7days.csv"

        self.states = [0, 1]

        self.transition_matrices = {}
        self.belief_states = {}

        # Cooperative sensing parameters
        self.detection_probability = (
            COOPERATIVE_DETECTION_PROBABILITY
        )

        self.false_alarm_probability = (
            COOPERATIVE_FALSE_ALARM_PROBABILITY
        )

        # Chronological training configuration
        self.training_days = 5
        self.testing_days = 2

        self.training_end = None
        self.testing_start = None

        self.rng = np.random.default_rng(RANDOM_SEED + 3)

        print(
            f"Cooperative Detection Probability: "
            f"{self.detection_probability:.5f}"
        )

        print(
            f"Cooperative False Alarm Probability: "
            f"{self.false_alarm_probability:.5f}"
        )

        self.load_dataset()
        self.split_dataset()
        self.train_hmm()
        self.initialize_beliefs()

        print("\nObservation-Aware HMM initialized!")

    def load_dataset(self):

        self.df = pd.read_csv(self.dataset_path)

        self.df["timestamp"] = pd.to_datetime(
            self.df["timestamp"]
        )

        print("Dataset loaded:", len(self.df), "records")

    def split_dataset(self):

        timestamps = pd.DatetimeIndex(
            sorted(self.df["timestamp"].unique())
        )

        total_days = self.df["timestamp"].dt.normalize().nunique()

        if total_days != self.training_days + self.testing_days:

            raise ValueError(
                "Expected 7 days of data for a 5-day/2-day split"
            )

        if len(timestamps) != 60480:

            raise ValueError(
                "Expected exactly 60,480 timestamps"
            )

        # First timestamp of day 6
        self.testing_start = timestamps[
            self.training_days * 8640
        ]

        self.training_end = self.testing_start

        # Only first five days are used for training
        self.training_df = self.df[
            self.df["timestamp"] < self.testing_start
        ].copy()

        # Last two days are reserved for evaluation
        self.testing_df = self.df[
            self.df["timestamp"] >= self.testing_start
        ].copy()

        print("\nCHRONOLOGICAL DATASET SPLIT")

        print(
            "Training records:",
            len(self.training_df)
        )

        print(
            "Testing records:",
            len(self.testing_df)
        )

        print(
            "Training period:",
            self.training_df["timestamp"].min(),
            "to",
            self.training_df["timestamp"].max()
        )

        print(
            "Testing period:",
            self.testing_df["timestamp"].min(),
            "to",
            self.testing_df["timestamp"].max()
        )

    def train_hmm(self):

        print("\nTraining HMM using first five days...\n")

        for channel_id in CHANNEL_IDS:

            channel_df = self.training_df[
                self.training_df["channel_id"] == channel_id
            ].sort_values("timestamp")

            observations = channel_df[
                "busy_state"
            ].to_numpy(dtype=int)

            transition_counts = np.zeros((2, 2))

            for i in range(len(observations) - 1):

                current = observations[i]
                next_state = observations[i + 1]

                transition_counts[current, next_state] += 1

            # Additive smoothing
            transition_counts += 1

            transition_matrix = (
                transition_counts /
                transition_counts.sum(axis=1, keepdims=True)
            )

            self.transition_matrices[channel_id] = (
                transition_matrix
            )

            print(channel_id)
            print(transition_matrix)
            print()

    def initialize_beliefs(self):

        for channel_id in CHANNEL_IDS:

            # Initialize using occupancy distribution
            # from training data only.
            channel_data = self.training_df[
                self.training_df["channel_id"] == channel_id
            ]["busy_state"]

            busy_probability = float(
                channel_data.mean()
            )

            idle_probability = 1 - busy_probability

            self.belief_states[channel_id] = np.array([
                idle_probability,
                busy_probability
            ])

    def observation_likelihood(self, observation):

        if observation == "IDLE":

            return np.array([
                1 - self.false_alarm_probability,
                1 - self.detection_probability
            ])

        elif observation == "BUSY":

            return np.array([
                self.false_alarm_probability,
                self.detection_probability
            ])

        else:

            raise ValueError(
                "Observation must be BUSY or IDLE"
            )

    def update_belief(self, channel_id, observation):

        if channel_id not in self.transition_matrices:

            raise ValueError(
                "Unknown channel: " + channel_id
            )

        previous_belief = self.belief_states[channel_id]

        transition_matrix = (
            self.transition_matrices[channel_id]
        )

        # Predict current hidden state
        predicted_belief = (
            previous_belief @ transition_matrix
        )

        # Observation likelihood
        likelihood = self.observation_likelihood(
            observation
        )

        # Bayesian correction
        updated_belief = predicted_belief * likelihood

        normalization = updated_belief.sum()

        if normalization <= 0:

            raise ValueError(
                "Invalid belief normalization"
            )

        updated_belief = (
            updated_belief / normalization
        )

        self.belief_states[channel_id] = updated_belief

        return updated_belief

    def predict_next_slot(self, channel_id):

        current_belief = self.belief_states[channel_id]

        transition_matrix = (
            self.transition_matrices[channel_id]
        )

        next_belief = (
            current_belief @ transition_matrix
        )

        return {
            "channel_id": channel_id,

            "probability_idle": float(
                next_belief[0]
            ),

            "probability_busy": float(
                next_belief[1]
            ),

            "predicted_state": (
                "BUSY"
                if next_belief[1] >= next_belief[0]
                else "IDLE"
            )
        }

    def process_observations(self, sensing_decisions):

        results = {}

        for channel_id in CHANNEL_IDS:

            observation = sensing_decisions[channel_id]

            updated_belief = self.update_belief(
                channel_id,
                observation
            )

            prediction = self.predict_next_slot(
                channel_id
            )

            results[channel_id] = {

                "observation": observation,

                "belief_idle": float(
                    updated_belief[0]
                ),

                "belief_busy": float(
                    updated_belief[1]
                ),

                "probability_idle": prediction[
                    "probability_idle"
                ],

                "probability_busy": prediction[
                    "probability_busy"
                ],

                "predicted_state": prediction[
                    "predicted_state"
                ]
            }

        return results

    def display_predictions(self, results):

        print("\n" + "=" * 75)

        print(" SPECTRIX OBSERVATION-AWARE HMM PREDICTIONS")

        print("=" * 75)

        for channel_id, result in results.items():

            print(
                f"{channel_id} | "
                f"Observation: {result['observation']} | "
                f"Current Busy Belief: "
                f"{result['belief_busy']:.4f} | "
                f"Next Busy Probability: "
                f"{result['probability_busy']:.4f} | "
                f"Prediction: {result['predicted_state']}"
            )


if __name__ == "__main__":

    from channel_simulator import ChannelSimulator
    from spectrum_sensing import SpectrumSensing

    channel_simulator = ChannelSimulator()

    sensing_engine = SpectrumSensing()

    hmm_predictor = HMMPredictor()

    # Test within the held-out evaluation period
    test_time = "2026-11-07 09:00:00"

    actual_status = channel_simulator.get_channel_status(
        test_time
    )

    sensing_reports = sensing_engine.perform_sensing(
        actual_status
    )

    final_decisions = sensing_engine.cooperative_sensing(
        sensing_reports
    )

    cooperative_decisions = {

        channel_id: result["decision"]

        for channel_id, result in final_decisions.items()
    }

    results = hmm_predictor.process_observations(
        cooperative_decisions
    )

    hmm_predictor.display_predictions(results)
