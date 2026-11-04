
from fuzzy_logic import FuzzyLogicEngine


class ASSMAllocator:

    def __init__(self, verbose=True):

        self.verbose = verbose

        if self.verbose:
            print("Initializing ASSM Channel Allocator...")

        self.fuzzy_engine = FuzzyLogicEngine()

        self.MAX_BUSY_PROBABILITY = 0.20
        self.MIN_SUITABILITY_SCORE = 40

        self.allocations = {}

        if self.verbose:
            print("ASSM Channel Allocator initialized!")

    def allocate_channels(
        self,
        requesting_users,
        sensing_decisions,
        hmm_predictions,
        channel_conditions
    ):

        self.allocations = {}

        allocated_channels = set()

        if self.verbose:
            print("\n" + "=" * 65)
            print(" SPECTRIX ASSM CHANNEL ALLOCATION")
            print("=" * 65)

        for user_id in requesting_users:

            candidate_channels = []

            if self.verbose:
                print(f"\nEvaluating channels for {user_id}...")

            for channel_id in sensing_decisions:

                if channel_id in allocated_channels:
                    continue

                if sensing_decisions[channel_id] != "IDLE":
                    continue

                busy_probability = hmm_predictions[
                    channel_id
                ]["probability_busy"]

                if busy_probability > self.MAX_BUSY_PROBABILITY:
                    continue

                snr = channel_conditions[channel_id]["snr"]

                interference = channel_conditions[
                    channel_id
                ]["interference"]

                fuzzy_result = self.fuzzy_engine.evaluate_channel(
                    snr,
                    busy_probability,
                    interference
                )

                suitability_score = fuzzy_result[
                    "suitability_score"
                ]

                if suitability_score < self.MIN_SUITABILITY_SCORE:
                    continue

                candidate_channels.append({
                    "channel_id": channel_id,
                    "busy_probability": busy_probability,
                    "snr": snr,
                    "interference": interference,
                    "suitability_score": suitability_score
                })

            if candidate_channels:

                best_channel = max(
                    candidate_channels,
                    key=lambda item: item["suitability_score"]
                )

                selected_channel = best_channel["channel_id"]

                allocated_channels.add(selected_channel)

                self.allocations[user_id] = {
                    "status": "ALLOCATED",
                    "channel_id": selected_channel,
                    "suitability_score": best_channel[
                        "suitability_score"
                    ],
                    "busy_probability": best_channel[
                        "busy_probability"
                    ]
                }

                if self.verbose:
                    print(
                        f"{user_id} -> {selected_channel} "
                        f"| Score: {best_channel['suitability_score']}"
                    )

            else:

                self.allocations[user_id] = {
                    "status": "NO_CHANNEL_AVAILABLE",
                    "channel_id": None,
                    "suitability_score": 0,
                    "busy_probability": None
                }

                if self.verbose:
                    print(
                        f"{user_id} -> No suitable channel available"
                    )

        return self.allocations

    def display_allocations(self):

        print("\n" + "=" * 65)
        print(" FINAL ASSM ALLOCATION RESULTS")
        print("=" * 65)

        for user_id, result in self.allocations.items():

            print(
                f"{user_id} | "
                f"Status: {result['status']} | "
                f"Channel: {result['channel_id']} | "
                f"Score: {result['suitability_score']}"
            )


if __name__ == "__main__":

    allocator = ASSMAllocator(verbose=True)

    sensing_decisions = {
        "CH01": "BUSY",
        "CH02": "BUSY",
        "CH03": "BUSY",
        "CH04": "BUSY",
        "CH05": "BUSY",
        "CH06": "IDLE",
        "CH07": "BUSY",
        "CH08": "IDLE"
    }

    hmm_predictions = {
        "CH01": {"probability_busy": 0.99},
        "CH02": {"probability_busy": 0.98},
        "CH03": {"probability_busy": 0.95},
        "CH04": {"probability_busy": 0.96},
        "CH05": {"probability_busy": 0.95},
        "CH06": {"probability_busy": 0.01},
        "CH07": {"probability_busy": 0.93},
        "CH08": {"probability_busy": 0.04}
    }

    channel_conditions = {
        "CH01": {"snr": 20, "interference": 10},
        "CH02": {"snr": 18, "interference": 20},
        "CH03": {"snr": 15, "interference": 30},
        "CH04": {"snr": 12, "interference": 40},
        "CH05": {"snr": 10, "interference": 50},
        "CH06": {"snr": 25, "interference": 10},
        "CH07": {"snr": 22, "interference": 15},
        "CH08": {"snr": 20, "interference": 20}
    }

    requesting_users = [
        "SU01", "SU02", "SU03", "SU04", "SU05"
    ]

    allocator.allocate_channels(
        requesting_users,
        sensing_decisions,
        hmm_predictions,
        channel_conditions
    )

    allocator.display_allocations()
