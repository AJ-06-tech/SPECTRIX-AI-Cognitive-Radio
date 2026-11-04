
# ============================================================
# SPECTRIX - STANDARD BASELINE CHANNEL ALLOCATOR
# ============================================================

from config import CHANNEL_IDS


class BaselineAllocator:

    def __init__(self, verbose=True):

        print("\nInitializing Standard Baseline Allocator...")

        self.channel_ids = CHANNEL_IDS

        self.allocations = {}

        self.verbose = verbose

        print("Allocation strategy: First Available Channel")
        print("HMM prediction: Disabled")
        print("Fuzzy logic: Disabled")

        print("Standard Baseline Allocator initialized!")

    # ========================================================
    # CHANNEL ALLOCATION
    # ========================================================

    def allocate_channels(
        self,
        requesting_users,
        sensing_decisions
    ):

        self.allocations = {}

        allocated_channels = set()

        for user_id in requesting_users:

            selected_channel = None

            # Check channels in fixed order.
            for channel_id in self.channel_ids:

                # Channel must be sensed IDLE.
                if sensing_decisions.get(channel_id) != "IDLE":
                    continue

                # Each channel can serve only one SU per slot.
                if channel_id in allocated_channels:
                    continue

                selected_channel = channel_id

                break

            # ------------------------------------------------
            # ALLOCATION RESULT
            # ------------------------------------------------

            if selected_channel is not None:

                allocated_channels.add(selected_channel)

                self.allocations[user_id] = {

                    "status": "ALLOCATED",

                    "channel_id": selected_channel,

                    "suitability_score": None,

                    "busy_probability": None

                }

                if self.verbose:

                    print(
                        f"{user_id} allocated to "
                        f"{selected_channel}"
                    )

            else:

                self.allocations[user_id] = {

                    "status": "NO_CHANNEL_AVAILABLE",

                    "channel_id": None,

                    "suitability_score": None,

                    "busy_probability": None

                }

                if self.verbose:

                    print(
                        f"{user_id}: No channel available"
                    )

        return self.allocations

    # ========================================================
    # DISPLAY ALLOCATION RESULTS
    # ========================================================

    def display_allocations(self):

        print("\n" + "=" * 60)

        print(" STANDARD BASELINE ALLOCATION RESULTS")

        print("=" * 60)

        if not self.allocations:

            print("No allocation results available.")

            return

        for user_id, result in self.allocations.items():

            print(
                f"{user_id} | "
                f"Status: {result['status']} | "
                f"Channel: {result['channel_id']}"
            )

        print("=" * 60)


# ============================================================
# INDEPENDENT TEST
# ============================================================

if __name__ == "__main__":

    allocator = BaselineAllocator()

    # Example cooperative sensing decisions.
    test_sensing_decisions = {

        "CH01": "BUSY",

        "CH02": "BUSY",

        "CH03": "IDLE",

        "CH04": "IDLE",

        "CH05": "BUSY",

        "CH06": "IDLE",

        "CH07": "IDLE",

        "CH08": "BUSY"

    }

    # Five secondary users requesting channels.
    test_requesting_users = [

        "SU01",

        "SU02",

        "SU03",

        "SU04",

        "SU05"

    ]

    results = allocator.allocate_channels(

        test_requesting_users,

        test_sensing_decisions

    )

    allocator.display_allocations()
