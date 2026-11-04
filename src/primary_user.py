
"""
SPECTRIX - Primary User Simulation

Part A: Predictive Rule-Based Spectrum Management

Responsibilities:
1. Create six Primary Users.
2. Assign one licensed channel to each PU.
3. Read occupancy from the existing CSV database.
4. Determine whether each PU is BUSY or IDLE.
5. Identify currently available licensed channels.
"""

import pandas as pd

from config import (
    DATABASE_FILE,
    LICENSED_CHANNELS,
    CHANNEL_TYPES,
    PRIMARY_USER_COUNT,
    SIMULATION_DAYS,
    SLOTS_PER_DAY
)


# ==================================================
# 1. PRIMARY USER CLASS
# ==================================================

class PrimaryUser:

    def __init__(self, user_id, channel_id):

        self.user_id = user_id
        self.channel_id = channel_id

        self.channel_type = CHANNEL_TYPES[channel_id]

        self.current_state = "UNKNOWN"

    def update_state(self, busy_state):

        if busy_state == 1:
            self.current_state = "BUSY"

        elif busy_state == 0:
            self.current_state = "IDLE"

        else:
            raise ValueError("Invalid occupancy state")

        return self.current_state

    def get_details(self):

        return {
            "user_id": self.user_id,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "current_state": self.current_state
        }


# ==================================================
# 2. PRIMARY USER MANAGER
# ==================================================

class PrimaryUserManager:

    def __init__(self):

        print("\nLoading Primary User Environment...")

        self.database = None
        self.occupancy = None

        self.primary_users = {}

        self.load_occupancy_database()

        self.create_primary_users()

    # ==============================================
    # 3. LOAD EXISTING OCCUPANCY DATABASE
    # ==============================================

    def load_occupancy_database(self):

        self.database = pd.read_csv(DATABASE_FILE)

        self.database["timestamp"] = pd.to_datetime(
            self.database["timestamp"]
        )

        # Convert long-format data into a time-channel matrix.
        self.occupancy = self.database.pivot(
            index="timestamp",
            columns="channel_id",
            values="busy_state"
        ).sort_index()

        # Validate licensed channels.
        for channel_id in LICENSED_CHANNELS:

            if channel_id not in self.occupancy.columns:
                raise ValueError(
                    f"Missing licensed channel: {channel_id}"
                )

        # Validate simulation duration.
        expected_slots = SIMULATION_DAYS * SLOTS_PER_DAY

        if len(self.occupancy) != expected_slots:
            raise ValueError(
                "Incorrect number of simulation time slots"
            )

        if self.occupancy[LICENSED_CHANNELS].isnull().any().any():
            raise ValueError(
                "Missing occupancy values in licensed channels"
            )

        print("Occupancy database loaded successfully!")

        print(
            f"Available time slots: {len(self.occupancy):,}"
        )

    # ==============================================
    # 4. CREATE SIX PRIMARY USERS
    # ==============================================

    def create_primary_users(self):

        print("\nCreating Primary Users...")

        for index, channel_id in enumerate(
            LICENSED_CHANNELS,
            start=1
        ):

            user_id = f"PU{index:02d}"

            self.primary_users[user_id] = PrimaryUser(
                user_id,
                channel_id
            )

            print(
                f"{user_id} assigned to {channel_id} "
                f"({CHANNEL_TYPES[channel_id]})"
            )

        if len(self.primary_users) != PRIMARY_USER_COUNT:
            raise ValueError("Primary User count mismatch")

        print("\nPrimary Users created successfully!")

    # ==============================================
    # 5. UPDATE PRIMARY USER STATES
    # ==============================================

    def update_all_states(self, slot_index):

        if not 0 <= slot_index < len(self.occupancy):
            raise IndexError("Invalid simulation slot")

        current_timestamp = self.occupancy.index[slot_index]

        states = {}

        for user_id, user in self.primary_users.items():

            busy_state = int(
                self.occupancy.iloc[slot_index][user.channel_id]
            )

            state = user.update_state(busy_state)

            states[user_id] = {
                "channel_id": user.channel_id,
                "channel_type": user.channel_type,
                "state": state
            }

        return current_timestamp, states

    # ==============================================
    # 6. GET AVAILABLE LICENSED CHANNELS
    # ==============================================

    def get_available_channels(self, slot_index):

        _, states = self.update_all_states(slot_index)

        available_channels = []

        for user_id, details in states.items():

            if details["state"] == "IDLE":

                available_channels.append(
                    details["channel_id"]
                )

        return available_channels

    # ==============================================
    # 7. DISPLAY PRIMARY USER STATUS
    # ==============================================

    def display_status(self, slot_index):

        timestamp, states = self.update_all_states(
            slot_index
        )

        print("\n======================================")
        print(" PRIMARY USER STATUS")
        print("======================================")

        print(f"Simulation time: {timestamp}")

        for user_id, details in states.items():

            print(
                f"{user_id} | "
                f"{details['channel_id']} | "
                f"{details['channel_type']} | "
                f"{details['state']}"
            )

        available = self.get_available_channels(slot_index)

        print("\nAvailable licensed channels:")

        if available:

            print(", ".join(available))

        else:

            print("No licensed channels currently idle.")

        return states


# ==================================================
# 8. MAIN EXECUTION
# ==================================================

if __name__ == "__main__":

    print("\n======================================")
    print(" SPECTRIX PRIMARY USER SIMULATOR")
    print("======================================")

    manager = PrimaryUserManager()

    # Test selected times throughout one day.
    test_slots = {
        "Midnight": 0,
        "Early Morning": 6 * 60 * 60 // 10,
        "Morning": 9 * 60 * 60 // 10,
        "Afternoon": 14 * 60 * 60 // 10,
        "Evening": 19 * 60 * 60 // 10,
        "Late Night": 23 * 60 * 60 // 10
    }

    for period, slot_index in test_slots.items():

        print(f"\n\n######## {period.upper()} ########")

        manager.display_status(slot_index)

    print("\n======================================")
    print(" PRIMARY USER SIMULATION COMPLETED")
    print("======================================")
