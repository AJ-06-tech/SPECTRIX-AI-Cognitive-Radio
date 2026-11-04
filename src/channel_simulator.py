
import pandas as pd

from config import (
    RAW_DATA_DIR,
    CHANNEL_IDS,
    LICENSED_CHANNELS,
    UNLICENSED_CHANNELS
)


class ChannelSimulator:

    def __init__(self):

        print("Initializing Channel Simulator...")

        self.dataset_path = RAW_DATA_DIR / "occupancy_7days.csv"

        self.channel_data = None
        self.current_time = None

        self.load_channel_data()

        print("Channel Simulator initialized successfully!")

    def load_channel_data(self):

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Occupancy dataset not found: {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        # Validate required columns
        required_columns = [
            "timestamp",
            "channel_id",
            "busy_state"
        ]

        for column in required_columns:

            if column not in df.columns:

                raise ValueError(
                    f"Missing required column: {column}"
                )

        # Validate channel IDs
        if set(df["channel_id"].unique()) != set(CHANNEL_IDS):

            raise ValueError(
                "Dataset channel IDs do not match configuration"
            )

        # Convert long-format data into a timestamp-indexed matrix
        self.channel_data = df.pivot(
            index="timestamp",
            columns="channel_id",
            values="busy_state"
        )

        # Ensure consistent channel ordering
        self.channel_data = self.channel_data.reindex(
            columns=CHANNEL_IDS
        )

        # Sort timestamps chronologically
        self.channel_data = self.channel_data.sort_index()

        # Validate missing values
        if self.channel_data.isnull().values.any():

            raise ValueError(
                "Occupancy dataset contains missing channel states"
            )

        # Validate binary occupancy values
        if not self.channel_data.isin([0, 1]).values.all():

            raise ValueError(
                "Occupancy states must be either 0 or 1"
            )

        print("\nOccupancy dataset loaded successfully.")

        print(
            "Total timestamps:",
            len(self.channel_data)
        )

        print(
            "Total channels:",
            len(self.channel_data.columns)
        )

        print(
            "Total simulation slots:",
            len(self.channel_data) 
        )

        print(
            "Time range:",
            self.channel_data.index.min(),
            "to",
            self.channel_data.index.max()
        )

    def get_channel_status(self, simulation_time):

        simulation_time = pd.Timestamp(simulation_time)

        if simulation_time not in self.channel_data.index:

            raise ValueError(
                f"No channel data available for {simulation_time}"
            )

        self.current_time = simulation_time

        # Direct timestamp lookup instead of filtering the entire CSV
        slot_data = self.channel_data.loc[simulation_time]

        channel_status = {}

        for channel_id in CHANNEL_IDS:

            busy_state = int(slot_data[channel_id])

            if channel_id in LICENSED_CHANNELS:

                if busy_state == 1:
                    status = "OCCUPIED"
                else:
                    status = "AVAILABLE"

            elif channel_id in UNLICENSED_CHANNELS:

                if busy_state == 1:
                    status = "EXTERNAL_ACTIVITY"
                else:
                    status = "AVAILABLE"

            else:

                status = "UNKNOWN"

            channel_status[channel_id] = {

                "status": status,

                "busy_state": busy_state,

                "licensed": (
                    channel_id in LICENSED_CHANNELS
                )
            }

        return channel_status

    def display_channel_status(self, simulation_time):

        status_data = self.get_channel_status(
            simulation_time
        )

        print("\n" + "=" * 55)

        print(" SPECTRIX CHANNEL STATUS")

        print("=" * 55)

        print(f"Simulation time: {simulation_time}\n")

        for channel_id, details in status_data.items():

            print(
                f"{channel_id} | "
                f"{details['status']} | "
                f"Licensed: {details['licensed']}"
            )

        available_channels = [

            channel_id

            for channel_id, details in status_data.items()

            if details["status"] == "AVAILABLE"
        ]

        print(
            "\nAvailable channels:",
            available_channels
        )

        return status_data

    def get_all_timestamps(self):

        return self.channel_data.index

    def get_total_slots(self):

        return len(self.channel_data)


if __name__ == "__main__":

    simulator = ChannelSimulator()

    test_times = [

        "2026-11-02 00:00:00",

        "2026-11-02 06:00:00",

        "2026-11-02 09:00:00",

        "2026-11-02 14:00:00",

        "2026-11-02 19:00:00",

        "2026-11-02 23:00:00"
    ]

    for test_time in test_times:

        simulator.display_channel_status(
            test_time
        )

    print("\n" + "=" * 55)

    print("CHANNEL SIMULATOR VALIDATION")

    print("=" * 55)

    print(
        "Total simulation slots:",
        simulator.get_total_slots()
    )

    print(
        "First timestamp:",
        simulator.get_all_timestamps()[0]
    )

    print(
        "Last timestamp:",
        simulator.get_all_timestamps()[-1]
    )

    print("\nChannel Simulator validation completed!")
