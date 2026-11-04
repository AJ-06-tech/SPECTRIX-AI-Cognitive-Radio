"""
SPECTRIX - Robustness Testing Spectrum Sensing

Supports variable secondary user counts.
Original spectrum_sensing.py remains unchanged.
"""

import numpy as np

from config import CHANNEL_IDS, RANDOM_SEED
from robustness_config import get_experiment_config


class RobustnessSpectrumSensing:

    def __init__(self, user_count, verbose=False):

        self.settings = get_experiment_config(user_count)

        self.user_ids = self.settings["secondary_user_ids"]

        self.channel_ids = self.settings["channel_ids"]

        self.detection_probability = (
            self.settings["individual_detection_probability"]
        )

        self.false_alarm_probability = (
            self.settings["individual_false_alarm_probability"]
        )

        self.required_votes = (
            self.settings["required_busy_votes"]
        )

        self.rng = np.random.default_rng(RANDOM_SEED + 2)

        self.verbose = verbose

        if self.verbose:
            print("Robustness Spectrum Sensing initialized.")
            print("Secondary users:", len(self.user_ids))
            print("Required BUSY votes:", self.required_votes)

    def sense_channel(self, actual_busy_state):

        random_value = self.rng.random()

        if actual_busy_state == 1:

            if random_value < self.detection_probability:
                return "SENSED_BUSY"

            return "SENSED_IDLE"

        if actual_busy_state == 0:

            if random_value < self.false_alarm_probability:
                return "SENSED_BUSY"

            return "SENSED_IDLE"

        raise ValueError("Actual channel state must be 0 or 1.")

    def perform_sensing(self, channel_status):

        sensing_reports = {}

        for user_id in self.user_ids:

            sensing_reports[user_id] = {}

            for channel_id in self.channel_ids:

                actual_state = channel_status[channel_id]["busy_state"]

                sensing_reports[user_id][channel_id] = (
                    self.sense_channel(actual_state)
                )

        return sensing_reports

    def cooperative_sensing(self, sensing_reports):

        if not sensing_reports:
            raise ValueError("Sensing reports cannot be empty.")

        if set(sensing_reports) != set(self.user_ids):
            raise ValueError(
                "Sensing reports do not match configured users."
            )

        final_decisions = {}

        total_users = len(sensing_reports)

        required_votes = (total_users // 2) + 1

        for channel_id in self.channel_ids:

            busy_votes = sum(
                1
                for user_id in self.user_ids
                if sensing_reports[user_id][channel_id] == "SENSED_BUSY"
            )

            idle_votes = total_users - busy_votes

            decision = (
                "BUSY"
                if busy_votes >= required_votes
                else "IDLE"
            )

            final_decisions[channel_id] = {
                "decision": decision,
                "busy_votes": busy_votes,
                "idle_votes": idle_votes
            }

        return final_decisions


if __name__ == "__main__":

    for count in [3, 5, 8, 10]:

        engine = RobustnessSpectrumSensing(
            user_count=count,
            verbose=True
        )

        print(
            f"Validated {count} users | "
            f"Threshold: {engine.required_votes}"
        )