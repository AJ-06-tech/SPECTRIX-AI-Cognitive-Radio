
import numpy as np

from config import CHANNEL_IDS, RANDOM_SEED


class ChannelQualityModel:

    def __init__(self):

        print("Initializing Channel Quality Model...")

        self.rng = np.random.default_rng(RANDOM_SEED + 5)

        # Illustrative baseline SNR values in dB
        self.base_snr = {
            "CH01": 18,
            "CH02": 20,
            "CH03": 22,
            "CH04": 21,
            "CH05": 24,
            "CH06": 25,
            "CH07": 20,
            "CH08": 22
        }

        print("Channel Quality Model initialized!")

    def generate_conditions(self, channel_status):

        conditions = {}

        for channel_id in CHANNEL_IDS:

            status_info = channel_status[channel_id]

            status = status_info["status"]

            # Generate small random signal variations
            snr_variation = self.rng.uniform(-3, 3)

            snr = self.base_snr[channel_id] + snr_variation

            # Estimate interference based on channel status
            if status == "OCCUPIED":

                interference = self.rng.uniform(85, 100)
                snr -= self.rng.uniform(5, 10)

            elif status == "EXTERNAL_ACTIVITY":

                interference = self.rng.uniform(40, 80)
                snr -= self.rng.uniform(2, 6)

            else:

                # Channel is available
                if channel_id in ["CH07", "CH08"]:

                    interference = self.rng.uniform(15, 40)

                else:

                    interference = self.rng.uniform(5, 25)

            # Keep values within fuzzy logic input ranges
            snr = float(np.clip(snr, -10, 30))

            interference = float(
                np.clip(interference, 0, 100)
            )

            conditions[channel_id] = {
                "snr": round(snr, 2),
                "interference": round(interference, 2)
            }

        return conditions

    def display_conditions(self, conditions):

        print("\n" + "=" * 65)
        print(" SPECTRIX CHANNEL QUALITY REPORT")
        print("=" * 65)

        print(
            f"{'Channel':<12}"
            f"{'SNR (dB)':<18}"
            f"{'Interference (%)':<20}"
        )

        print("-" * 65)

        for channel_id, values in conditions.items():

            print(
                f"{channel_id:<12}"
                f"{values['snr']:<18}"
                f"{values['interference']:<20}"
            )


if __name__ == "__main__":

    from channel_simulator import ChannelSimulator

    simulator = ChannelSimulator()

    quality_model = ChannelQualityModel()

    test_time = "2026-11-02 09:00:00"

    channel_status = simulator.get_channel_status(
        test_time
    )

    conditions = quality_model.generate_conditions(
        channel_status
    )

    quality_model.display_conditions(conditions)
