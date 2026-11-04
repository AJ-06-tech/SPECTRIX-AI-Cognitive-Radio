
import numpy as np

from config import (
    PACKET_SIZE_BYTES,
    MAX_TRANSMISSION_ATTEMPTS,
    RANDOM_SEED
)


class TransmissionSimulator:

    def __init__(self):

        print("Initializing Transmission Simulator...")

        self.rng = np.random.default_rng(RANDOM_SEED + 4)

        # Global statistics
        self.total_transmission_attempts = 0
        self.successful_transmissions = 0
        self.failed_transmissions = 0
        self.failed_attempts = 0
        self.total_packets_sent = 0
        self.total_bytes_transmitted = 0
        self.total_handovers = 0
        self.total_collisions = 0

        self.user_statistics = {}

        print("Packet size:", PACKET_SIZE_BYTES, "bytes")
        print("Maximum attempts:", MAX_TRANSMISSION_ATTEMPTS)
        print("Transmission Simulator initialized!")

    def initialize_user(self, user_id):

        if user_id not in self.user_statistics:

            self.user_statistics[user_id] = {
                "attempts": 0,
                "successful": 0,
                "failed": 0,
                "failed_attempts": 0,
                "packets_sent": 0,
                "bytes_sent": 0,
                "handovers": 0,
                "collisions": 0,
                "current_channel": None
            }

    def transmit_packet(
        self,
        user_id,
        channel_id,
        channel_busy_state,
        channel_snr,
        attempt_number=1,
        simulation_time=None
    ):

        self.initialize_user(user_id)

        stats = self.user_statistics[user_id]

        stats["current_channel"] = channel_id

        # Every transmission attempt must be counted,
        # including attempts blocked by an occupied channel.
        self.total_transmission_attempts += 1
        stats["attempts"] += 1

        # Safety check: channel must be idle
        if channel_busy_state == 1:

            self.total_collisions += 1
            stats["collisions"] += 1

            self.failed_attempts += 1
            stats["failed_attempts"] += 1

            self.release_channel(user_id)

            return {
                "status": "COLLISION",
                "bytes_transmitted": 0,
                "channel": channel_id,
                "attempt_number": attempt_number,
                "retry_required": (
                    attempt_number < MAX_TRANSMISSION_ATTEMPTS
                ),
                "packet_failed": (
                    attempt_number >= MAX_TRANSMISSION_ATTEMPTS
                ),
                "simulation_time": simulation_time
            }

        # Calculate transmission success probability
        success_probability = float(
            np.clip(
                0.50 + (channel_snr / 60),
                0.10,
                0.98
            )
        )

        transmission_success = (
            self.rng.random() < success_probability
        )

        if transmission_success:

            self.successful_transmissions += 1

            self.total_packets_sent += 1

            self.total_bytes_transmitted += PACKET_SIZE_BYTES

            stats["successful"] += 1
            stats["packets_sent"] += 1
            stats["bytes_sent"] += PACKET_SIZE_BYTES

            self.release_channel(user_id)

            return {
                "status": "SUCCESS",
                "bytes_transmitted": PACKET_SIZE_BYTES,
                "channel": channel_id,
                "attempt_number": attempt_number,
                "retry_required": False,
                "packet_failed": False,
                "simulation_time": simulation_time
            }

        # Transmission attempt failed
        self.failed_attempts += 1
        stats["failed_attempts"] += 1

        self.release_channel(user_id)

        # Check whether retry limit has been reached
        if attempt_number >= MAX_TRANSMISSION_ATTEMPTS:

            self.failed_transmissions += 1
            stats["failed"] += 1

            return {
                "status": "FAILED",
                "bytes_transmitted": 0,
                "channel": channel_id,
                "attempt_number": attempt_number,
                "retry_required": False,
                "packet_failed": True,
                "simulation_time": simulation_time
            }

        return {
            "status": "RETRY_REQUIRED",
            "bytes_transmitted": 0,
            "channel": channel_id,
            "attempt_number": attempt_number,
            "retry_required": True,
            "packet_failed": False,
            "simulation_time": simulation_time
        }

    def release_channel(self, user_id):

        if user_id in self.user_statistics:

            self.user_statistics[user_id]["current_channel"] = None

    def perform_handover(
        self,
        user_id,
        new_channel,
        channel_status=None
    ):

        self.initialize_user(user_id)

        # Handover must not select an occupied channel
        if channel_status is not None:

            if new_channel not in channel_status:

                return False

            if channel_status[new_channel]["status"] != "AVAILABLE":

                return False

        self.total_handovers += 1

        self.user_statistics[user_id]["handovers"] += 1

        self.user_statistics[user_id]["current_channel"] = new_channel

        return True

    def get_statistics(self):

        return {
            "total_transmission_attempts":
                self.total_transmission_attempts,

            "successful_transmissions":
                self.successful_transmissions,

            "failed_transmissions":
                self.failed_transmissions,

            "failed_attempts":
                self.failed_attempts,

            "total_packets_sent":
                self.total_packets_sent,

            "total_bytes_transmitted":
                self.total_bytes_transmitted,

            "total_collisions":
                self.total_collisions,

            "total_handovers":
                self.total_handovers
        }

    def display_statistics(self):

        print("\n" + "=" * 60)
        print(" SPECTRIX TRANSMISSION STATISTICS")
        print("=" * 60)

        print(
            "Total transmission attempts:",
            self.total_transmission_attempts
        )

        print(
            "Successful transmissions:",
            self.successful_transmissions
        )

        print(
            "Failed packets:",
            self.failed_transmissions
        )

        print(
            "Failed attempts:",
            self.failed_attempts
        )

        print(
            "Total packets sent:",
            self.total_packets_sent
        )

        print(
            "Total bytes transmitted:",
            self.total_bytes_transmitted
        )

        print(
            "Total collisions:",
            self.total_collisions
        )

        print(
            "Total handovers:",
            self.total_handovers
        )

        if self.total_transmission_attempts > 0:

            attempt_success_rate = (
                self.successful_transmissions /
                self.total_transmission_attempts
            ) * 100

            print(
                "Attempt success rate:",
                round(attempt_success_rate, 2),
                "%"
            )

        print("\nIndividual User Statistics:")

        for user_id, stats in self.user_statistics.items():

            print(user_id, stats)


if __name__ == "__main__":

    simulator = TransmissionSimulator()

    print("\nTEST 1: Available channel")

    result1 = simulator.transmit_packet(
        user_id="SU01",
        channel_id="CH06",
        channel_busy_state=0,
        channel_snr=25,
        attempt_number=1,
        simulation_time="2026-11-07 09:00:00"
    )

    print("Result:", result1)

    print("\nTEST 2: Occupied channel")

    result2 = simulator.transmit_packet(
        user_id="SU02",
        channel_id="CH03",
        channel_busy_state=1,
        channel_snr=20,
        attempt_number=1,
        simulation_time="2026-11-07 09:00:00"
    )

    print("Result:", result2)

    print("\nTEST 3: Handover to available channel")

    available_status = {
        "CH08": {
            "status": "AVAILABLE"
        }
    }

    handover_result = simulator.perform_handover(
        user_id="SU02",
        new_channel="CH08",
        channel_status=available_status
    )

    print("Handover successful:", handover_result)

    simulator.display_statistics()
