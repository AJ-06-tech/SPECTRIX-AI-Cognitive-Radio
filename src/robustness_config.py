"""
SPECTRIX - Robustness Testing Configuration

Separate settings for scalability and robustness experiments.
Does not modify the original Part A configuration.
"""

from math import comb

from config import (
    RANDOM_SEED,
    CHANNEL_IDS,
    LICENSED_CHANNELS,
    UNLICENSED_CHANNELS,
    SLOT_DURATION_SECONDS,
    PACKET_SIZE_BYTES,
    MAX_TRANSMISSION_ATTEMPTS,
    SENSING_DETECTION_PROBABILITY,
    SENSING_FALSE_ALARM_PROBABILITY
)


# ============================================================
# 1. SCALABILITY EXPERIMENTS
# ============================================================

SCALABILITY_USER_COUNTS = [3, 5, 8, 10]

TRAINING_DAYS = 5
TESTING_DAYS = 2


# ============================================================
# 2. FIXED NETWORK SETTINGS
# ============================================================

TOTAL_CHANNELS = len(CHANNEL_IDS)

SLOT_SECONDS = SLOT_DURATION_SECONDS

PACKET_BYTES = PACKET_SIZE_BYTES

MAX_ATTEMPTS = MAX_TRANSMISSION_ATTEMPTS


# ============================================================
# 3. INDIVIDUAL SENSING SETTINGS
# ============================================================

INDIVIDUAL_DETECTION_PROBABILITY = (
    SENSING_DETECTION_PROBABILITY
)

INDIVIDUAL_FALSE_ALARM_PROBABILITY = (
    SENSING_FALSE_ALARM_PROBABILITY
)


# ============================================================
# 4. COOPERATIVE SENSING CALCULATIONS
# ============================================================

def majority_vote_threshold(user_count):

    if user_count < 1:
        raise ValueError("User count must be at least 1.")

    return (user_count // 2) + 1


def calculate_cooperative_probability(
    user_count,
    individual_probability
):

    required_votes = majority_vote_threshold(user_count)

    if not 0 <= individual_probability <= 1:
        raise ValueError(
            "Individual probability must be between 0 and 1."
        )

    probability = sum(
        comb(user_count, votes)
        * individual_probability ** votes
        * (1 - individual_probability) ** (user_count - votes)
        for votes in range(required_votes, user_count + 1)
    )

    return probability


# ============================================================
# 5. EXPERIMENT CONFIGURATION
# ============================================================

def get_experiment_config(user_count):

    if user_count not in SCALABILITY_USER_COUNTS:
        raise ValueError(
            f"Unsupported user count: {user_count}. "
            f"Choose from {SCALABILITY_USER_COUNTS}."
        )

    user_ids = [
        f"SU{i:02d}"
        for i in range(1, user_count + 1)
    ]

    required_votes = majority_vote_threshold(user_count)

    cooperative_pd = calculate_cooperative_probability(
        user_count,
        INDIVIDUAL_DETECTION_PROBABILITY
    )

    cooperative_pfa = calculate_cooperative_probability(
        user_count,
        INDIVIDUAL_FALSE_ALARM_PROBABILITY
    )

    return {
        "secondary_user_count": user_count,
        "secondary_user_ids": user_ids,

        "total_channels": TOTAL_CHANNELS,
        "channel_ids": CHANNEL_IDS,
        "licensed_channels": LICENSED_CHANNELS,
        "unlicensed_channels": UNLICENSED_CHANNELS,

        "training_days": TRAINING_DAYS,
        "testing_days": TESTING_DAYS,

        "slot_duration_seconds": SLOT_SECONDS,
        "packet_size_bytes": PACKET_BYTES,
        "max_transmission_attempts": MAX_ATTEMPTS,

        "individual_detection_probability":
            INDIVIDUAL_DETECTION_PROBABILITY,

        "individual_false_alarm_probability":
            INDIVIDUAL_FALSE_ALARM_PROBABILITY,

        "required_busy_votes": required_votes,

        "cooperative_detection_probability": cooperative_pd,

        "cooperative_false_alarm_probability": cooperative_pfa,

        "random_seed": RANDOM_SEED
    }


# ============================================================
# 6. DISPLAY EXPERIMENT SETTINGS
# ============================================================

def display_scalability_plan():

    print("\n" + "=" * 75)
    print(" SPECTRIX - SECONDARY USER SCALABILITY CONFIGURATION")
    print("=" * 75)

    print("\nIndividual sensing performance:")
    print(
        f"Detection probability: "
        f"{INDIVIDUAL_DETECTION_PROBABILITY:.4f}"
    )
    print(
        f"False alarm probability: "
        f"{INDIVIDUAL_FALSE_ALARM_PROBABILITY:.4f}"
    )

    print("\nExperiment configurations:\n")

    print(
        f"{'Users':<10}"
        f"{'Required Votes':<18}"
        f"{'Cooperative Pd':<20}"
        f"{'Cooperative Pfa':<20}"
    )

    print("-" * 68)

    for user_count in SCALABILITY_USER_COUNTS:

        settings = get_experiment_config(user_count)

        print(
            f"{user_count:<10}"
            f"{settings['required_busy_votes']:<18}"
            f"{settings['cooperative_detection_probability']:<20.8f}"
            f"{settings['cooperative_false_alarm_probability']:<20.8f}"
        )

    print("\nOriginal config.py remains unchanged.")


if __name__ == "__main__":

    display_scalability_plan()