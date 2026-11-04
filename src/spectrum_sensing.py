
import numpy as np

from config import (
    CHANNEL_IDS,
    SECONDARY_USER_IDS,
    RANDOM_SEED
)


class SpectrumSensing:

    def __init__(self):

        print("Initializing Spectrum Sensing Engine...")

        self.detection_probability = 0.90
        self.false_alarm_probability = 0.05

        self.rng = np.random.default_rng(RANDOM_SEED + 2)

        print("Detection probability:", self.detection_probability)
        print("False alarm probability:", self.false_alarm_probability)
        print("Secondary Users:", len(SECONDARY_USER_IDS))

        print("Spectrum Sensing Engine initialized!")

    def sense_channel(self, actual_busy_state):

        random_value = self.rng.random()

        if actual_busy_state == 1:

            if random_value < self.detection_probability:
                return "SENSED_BUSY"

            return "SENSED_IDLE"

        else:

            if random_value < self.false_alarm_probability:
                return "SENSED_BUSY"

            return "SENSED_IDLE"

    def perform_sensing(self, channel_status):

        sensing_reports = {}

        for user_id in SECONDARY_USER_IDS:

            sensing_reports[user_id] = {}

            for channel_id in CHANNEL_IDS:

                actual_busy_state = channel_status[channel_id]["busy_state"]

                observation = self.sense_channel(actual_busy_state)

                sensing_reports[user_id][channel_id] = observation

        return sensing_reports

    def cooperative_sensing(self, sensing_reports):

        final_decisions = {}

        total_users = len(sensing_reports)

        required_votes = (total_users // 2) + 1

        for channel_id in CHANNEL_IDS:

            busy_votes = 0
            idle_votes = 0

            for user_id in sensing_reports:

                observation = sensing_reports[user_id][channel_id]

                if observation == "SENSED_BUSY":
                    busy_votes += 1
                else:
                    idle_votes += 1

            if busy_votes >= required_votes:
                decision = "BUSY"
            else:
                decision = "IDLE"

            final_decisions[channel_id] = {
                "decision": decision,
                "busy_votes": busy_votes,
                "idle_votes": idle_votes
            }

        return final_decisions

    def display_results(self, sensing_reports, final_decisions):

        print("\n" + "=" * 65)
        print(" SPECTRIX COOPERATIVE SPECTRUM SENSING")
        print("=" * 65)

        print("\nINDIVIDUAL SECONDARY USER REPORTS\n")

        for user_id, reports in sensing_reports.items():

            print(user_id, end=" | ")

            for channel_id in CHANNEL_IDS:

                print(
                    f"{channel_id}: {reports[channel_id]}",
                    end=" | "
                )

            print()

        print("\nCOOPERATIVE SENSING DECISIONS\n")

        for channel_id, result in final_decisions.items():

            print(
                f"{channel_id} | "
                f"Decision: {result['decision']} | "
                f"Busy votes: {result['busy_votes']} | "
                f"Idle votes: {result['idle_votes']}"
            )


if __name__ == "__main__":

    from channel_simulator import ChannelSimulator

    channel_simulator = ChannelSimulator()
    sensing_engine = SpectrumSensing()

    test_time = "2026-11-02 09:00:00"

    actual_channel_status = channel_simulator.get_channel_status(
        test_time
    )

    sensing_reports = sensing_engine.perform_sensing(
        actual_channel_status
    )

    final_decisions = sensing_engine.cooperative_sensing(
        sensing_reports
    )

    sensing_engine.display_results(
        sensing_reports,
        final_decisions
    )
