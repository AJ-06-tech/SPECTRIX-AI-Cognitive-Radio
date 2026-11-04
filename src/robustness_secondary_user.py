"""
SPECTRIX - Robustness Secondary User Environment

Creates a configurable number of secondary users
without modifying the original secondary_user.py.
"""

import numpy as np

from config import RANDOM_SEED
from secondary_user import SecondaryUser
from robustness_config import get_experiment_config


class RobustnessSecondaryUserEnvironment:

    def __init__(self, user_count, verbose=False):

        self.settings = get_experiment_config(user_count)

        self.user_ids = self.settings["secondary_user_ids"]

        # Independent reproducible RNG for each user.
        self.user_rngs = {
            user_id: np.random.default_rng(
                RANDOM_SEED + 1000 + int(user_id[2:])
            )
            for user_id in self.user_ids
        }

        self.users = {
            user_id: SecondaryUser(user_id)
            for user_id in self.user_ids
        }

        self.verbose = verbose

        if self.verbose:
            print("Robustness User Environment initialized.")
            print("Secondary users:", len(self.users))

    def generate_requests(self, time_period, simulation_time=None):

        generated_requests = []

        for user_id, user in self.users.items():

            request = user.generate_request(
                time_period,
                self.user_rngs[user_id],
                simulation_time
            )

            if request is not None:
                generated_requests.append(request)

        return generated_requests

    def get_requesting_users(self):

        return [
            user_id
            for user_id, user in self.users.items()
            if (
                user.pending_requests > 0
                and user.current_channel is None
            )
        ]

    def get_user(self, user_id):

        return self.users.get(user_id)

    def display_statistics(self):

        for user in self.users.values():
            print(user.get_statistics())


if __name__ == "__main__":

    print("\nSPECTRIX ROBUSTNESS TRAFFIC TEST")

    for count in [3, 5, 8, 10]:

        environment = RobustnessSecondaryUserEnvironment(
            user_count=count,
            verbose=True
        )

        requests = environment.generate_requests(
            "Morning",
            "2026-11-07 09:00:00"
        )

        print(
            f"Users: {count} | "
            f"Created users: {len(environment.users)} | "
            f"Requests generated in test slot: {len(requests)}"
        )

        assert len(environment.users) == count

        assert all(
            user_id in environment.user_rngs
            for user_id in environment.users
        )

        print("Traffic environment test: PASS")