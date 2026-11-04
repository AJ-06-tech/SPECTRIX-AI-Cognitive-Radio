
import numpy as np

# Supports both:
# python src/secondary_user.py
# and
# from src.secondary_user import SecondaryUser

try:
    from .config import (
        SECONDARY_USER_IDS,
        PACKET_SIZE_BYTES,
        MAX_TRANSMISSION_ATTEMPTS,
        RANDOM_SEED
    )
except ImportError:
    from config import (
        SECONDARY_USER_IDS,
        PACKET_SIZE_BYTES,
        MAX_TRANSMISSION_ATTEMPTS,
        RANDOM_SEED
    )


class SecondaryUser:

    REQUEST_PROBABILITIES = {
        "Midnight": 0.05,
        "Early Morning": 0.08,
        "Morning": 0.18,
        "Afternoon": 0.20,
        "Evening": 0.25,
        "Late Night": 0.10
    }

    def __init__(self, user_id):

        self.user_id = user_id

        self.current_channel = None
        self.current_state = "IDLE"

        self.total_requests = 0
        self.successful_allocations = 0
        self.pending_requests = 0
        self.total_packets_sent = 0
        self.failed_transmissions = 0
        self.total_retries = 0

        self.total_waiting_time = 0.0
        self.completed_request_count = 0

        self.request_queue = []

    # --------------------------------------------------
    # GENERATE PACKET REQUEST
    # --------------------------------------------------

    def generate_request(
        self,
        time_period,
        rng,
        simulation_time=None
    ):

        if time_period not in self.REQUEST_PROBABILITIES:

            raise ValueError(
                f"Unknown time period: {time_period}"
            )

        request_probability = self.REQUEST_PROBABILITIES[
            time_period
        ]

        if rng.random() < request_probability:

            request = {
                "user_id": self.user_id,
                "packet_size": PACKET_SIZE_BYTES,
                "status": "PENDING",
                "time_period": time_period,
                "created_at": simulation_time,
                "attempts": 0
            }

            self.request_queue.append(request)

            self.total_requests += 1
            self.pending_requests += 1

            self.current_state = "REQUESTING"

            return request

        if self.pending_requests > 0:

            self.current_state = "WAITING"

        else:

            self.current_state = "IDLE"

        return None

    # --------------------------------------------------
    # CHANNEL ALLOCATION
    # --------------------------------------------------

    def allocate_channel(self, channel_id):

        if self.pending_requests <= 0:
            return False

        if self.current_channel is not None:
            return False

        if not self.request_queue:
            return False

        self.current_channel = channel_id

        self.successful_allocations += 1

        request = self.request_queue[0]

        # Count retry if packet was previously attempted
        if request["attempts"] > 0:

            self.total_retries += 1

        request["attempts"] += 1
        request["status"] = "TRANSMITTING"

        self.current_state = "TRANSMITTING"

        return True

    # --------------------------------------------------
    # COMPLETE TRANSMISSION
    # --------------------------------------------------

    def complete_transmission(
        self,
        transmission_result,
        completion_time=None
    ):

        if self.pending_requests <= 0:
            return False

        if not self.request_queue:
            return False

        request = self.request_queue[0]

        # Accept the new transmission result dictionary
        # and retain compatibility with True/False calls.
        if isinstance(transmission_result, dict):

            status = transmission_result.get("status")

        elif transmission_result is True:

            status = "SUCCESS"

        elif transmission_result is False:

            status = "RETRY_REQUIRED"

        else:

            raise ValueError(
                "Invalid transmission result."
            )

        # CASE 1: SUCCESS
        if status == "SUCCESS":

            request["status"] = "COMPLETED"

            # Calculate packet waiting time
            if (
                request["created_at"] is not None
                and completion_time is not None
            ):

                created_at = np.datetime64(
                    request["created_at"]
                )

                completed_at = np.datetime64(
                    completion_time
                )

                waiting_time = float(
                    (completed_at - created_at)
                    / np.timedelta64(1, "s")
                )

                self.total_waiting_time += max(
                    0.0, waiting_time
                )

            self.completed_request_count += 1
            self.total_packets_sent += 1

            self.request_queue.pop(0)

            self.pending_requests -= 1

            self.release_channel()

            return True

        # CASE 2: COLLISION OR RETRY REQUIRED
        if status in (
            "COLLISION",
            "RETRY_REQUIRED"
        ):

            packet_failed = False

            if isinstance(transmission_result, dict):

                packet_failed = transmission_result.get(
                    "packet_failed",
                    False
                )

            if not packet_failed:

                request["status"] = "PENDING"

                self.release_channel()

                return False

            status = "FAILED"

        # CASE 3: MAXIMUM ATTEMPTS REACHED
        if status == "FAILED":

            request["status"] = "FAILED"

            self.failed_transmissions += 1

            self.request_queue.pop(0)

            self.pending_requests -= 1

            self.release_channel()

            return False

        raise ValueError(
            f"Unknown transmission status: {status}"
        )

    # --------------------------------------------------
    # RELEASE CHANNEL
    # --------------------------------------------------

    def release_channel(self):

        self.current_channel = None

        if self.request_queue:

            self.request_queue[0]["status"] = "PENDING"

        if self.pending_requests > 0:

            self.current_state = "WAITING"

        else:

            self.current_state = "IDLE"

    # --------------------------------------------------
    # USER STATISTICS
    # --------------------------------------------------

    def get_statistics(self):

        average_waiting_time = 0.0

        if self.completed_request_count > 0:

            average_waiting_time = (
                self.total_waiting_time
                / self.completed_request_count
            )

        return {
            "user_id": self.user_id,
            "current_channel": self.current_channel,
            "current_state": self.current_state,
            "total_requests": self.total_requests,
            "successful_allocations": self.successful_allocations,
            "pending_requests": self.pending_requests,
            "total_packets_sent": self.total_packets_sent,
            "failed_transmissions": self.failed_transmissions,
            "total_retries": self.total_retries,
            "average_waiting_time": round(
                average_waiting_time,
                3
            )
        }


# ======================================================
# SECONDARY USER ENVIRONMENT
# ======================================================

class SecondaryUserEnvironment:

    def __init__(self):

        print("Initializing Secondary User Environment...")

        self.rng = np.random.default_rng(
            RANDOM_SEED + 1
        )

        self.users = {
            user_id: SecondaryUser(user_id)
            for user_id in SECONDARY_USER_IDS
        }

        print(
            "All Secondary Users created successfully!"
        )

    # --------------------------------------------------
    # GENERATE REQUESTS FOR ALL USERS
    # --------------------------------------------------

    def generate_requests(
        self,
        time_period,
        simulation_time=None
    ):

        generated_requests = []

        for user in self.users.values():

            request = user.generate_request(
                time_period,
                self.rng,
                simulation_time
            )

            if request is not None:

                generated_requests.append(request)

        return generated_requests

    # --------------------------------------------------
    # GET USERS WAITING FOR CHANNEL ALLOCATION
    # --------------------------------------------------

    def get_requesting_users(self):

        return [
            user_id
            for user_id, user in self.users.items()
            if (
                user.pending_requests > 0
                and user.current_channel is None
            )
        ]

    # --------------------------------------------------
    # GET INDIVIDUAL USER
    # --------------------------------------------------

    def get_user(self, user_id):

        return self.users.get(user_id)

    # --------------------------------------------------
    # DISPLAY CURRENT STATUS
    # --------------------------------------------------

    def display_status(
        self,
        simulation_time,
        generated_requests
    ):

        print("\n" + "=" * 65)

        print(
            " SPECTRIX SECONDARY USER REQUEST STATUS"
        )

        print("=" * 65)

        print("Simulation time:", simulation_time)
        print("Total users:", len(self.users))
        print(
            "Requests generated:",
            len(generated_requests)
        )

        for user in self.users.values():

            stats = user.get_statistics()

            print(
                stats["user_id"],
                "|",
                stats["current_state"],
                "| Requests:",
                stats["total_requests"],
                "| Pending:",
                stats["pending_requests"],
                "| Packets sent:",
                stats["total_packets_sent"]
            )

    # --------------------------------------------------
    # DISPLAY FINAL STATISTICS
    # --------------------------------------------------

    def display_statistics(self):

        print(
            "\nFINAL SECONDARY USER STATISTICS\n"
        )

        for user in self.users.values():

            print(user.get_statistics())


# ======================================================
# INDIVIDUAL MODULE TEST
# ======================================================

if __name__ == "__main__":

    environment = SecondaryUserEnvironment()

    test_times = [
        ("2026-11-02 00:00:00", "Midnight"),
        ("2026-11-02 06:00:00", "Early Morning"),
        ("2026-11-02 09:00:00", "Morning"),
        ("2026-11-02 14:00:00", "Afternoon"),
        ("2026-11-02 19:00:00", "Evening"),
        ("2026-11-02 23:00:00", "Late Night")
    ]

    for simulation_time, time_period in test_times:

        generated_requests = environment.generate_requests(
            time_period,
            simulation_time
        )

        environment.display_status(
            simulation_time,
            generated_requests
        )

    print(
        "\nSECONDARY USER SIMULATION COMPLETED"
    )

    environment.display_statistics()
