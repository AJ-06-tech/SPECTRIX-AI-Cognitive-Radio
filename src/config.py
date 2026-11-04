
"""
SPECTRIX - Central Simulation Configuration

Part A: Predictive Rule-Based Spectrum Management
Shared settings for Standard, Part A and Part B.
"""

from pathlib import Path


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
REPORT_DIR = OUTPUT_DIR / "reports"

DATABASE_FILE = RAW_DATA_DIR / "occupancy_7days.csv"


# ==========================================
# 2. SIMULATION SETTINGS
# ==========================================

PROJECT_NAME = "SPECTRIX"

RANDOM_SEED = 42

SIMULATION_DAYS = 7

SLOT_DURATION_SECONDS = 10

SLOTS_PER_DAY = (
    24 * 60 * 60 // SLOT_DURATION_SECONDS
)


# ==========================================
# 3. WIRELESS CHANNEL CONFIGURATION
# ==========================================

TOTAL_CHANNELS = 8

CHANNEL_IDS = [
    "CH01",
    "CH02",
    "CH03",
    "CH04",
    "CH05",
    "CH06",
    "CH07",
    "CH08"
]

CHANNEL_TYPES = {
    "CH01": "Broadcast",
    "CH02": "Broadcast",
    "CH03": "Cellular",
    "CH04": "Cellular",
    "CH05": "Business",
    "CH06": "Business",
    "CH07": "ISM",
    "CH08": "ISM"
}


# ==========================================
# 4. LICENSED AND UNLICENSED CHANNELS
# ==========================================

LICENSED_CHANNELS = [
    "CH01",
    "CH02",
    "CH03",
    "CH04",
    "CH05",
    "CH06"
]

UNLICENSED_CHANNELS = [
    "CH07",
    "CH08"
]


# ==========================================
# 5. PRIMARY USER CONFIGURATION
# ==========================================

# Primary users have priority on licensed channels.

PRIMARY_USER_COUNT = len(LICENSED_CHANNELS)


# ==========================================
# 6. SECONDARY USER CONFIGURATION
# ==========================================

# Initial experimental scenario.
# We will later test different user counts.

SECONDARY_USER_COUNT = 5

SECONDARY_USER_IDS = [
    f"SU{i:02d}"
    for i in range(1, SECONDARY_USER_COUNT + 1)
]


# ==========================================
# 7. NETWORK CONFIGURATION
# ==========================================

# Initial simulation parameters.
# These are configurable assumptions, not
# real-world measured network specifications.

PACKET_SIZE_BYTES = 1500

MAX_TRANSMISSION_ATTEMPTS = 3


# ==========================================
# 8. RANDOMIZATION
# ==========================================

# Same seed makes experiments reproducible.

import numpy as np

RANDOM_GENERATOR = np.random.default_rng(
    RANDOM_SEED
)


# ==========================================
# 9. DISPLAY CONFIGURATION
# ==========================================

def display_configuration():

    print("\n====================================")
    print(" SPECTRIX SIMULATION CONFIGURATION")
    print("====================================")

    print(f"Project: {PROJECT_NAME}")

    print(f"Simulation duration: {SIMULATION_DAYS} days")

    print(
        f"Sampling interval: "
        f"{SLOT_DURATION_SECONDS} seconds"
    )

    print(f"Total channels: {TOTAL_CHANNELS}")

    print(f"Licensed channels: {len(LICENSED_CHANNELS)}")

    print(f"Unlicensed channels: {len(UNLICENSED_CHANNELS)}")

    print(f"Primary users: {PRIMARY_USER_COUNT}")

    print(f"Secondary users: {SECONDARY_USER_COUNT}")

    print("\nChannel allocation:")

    for channel_id in CHANNEL_IDS:

        channel_type = CHANNEL_TYPES[channel_id]

        if channel_id in LICENSED_CHANNELS:
            category = "Licensed"
        else:
            category = "Unlicensed"

        print(
            f"{channel_id} | "
            f"{channel_type} | "
            f"{category}"
        )

    print("\nConfiguration loaded successfully!")


# ==========================================
# 10. TEST EXECUTION
# ==========================================

if __name__ == "__main__":

    display_configuration()

# Cooperative Spectrum Sensing Configuration

SENSING_DETECTION_PROBABILITY = 0.90
SENSING_FALSE_ALARM_PROBABILITY = 0.05

COOPERATIVE_DETECTION_PROBABILITY = 0.99144
COOPERATIVE_FALSE_ALARM_PROBABILITY = 0.00116

COOPERATIVE_SENSOR_COUNT = SECONDARY_USER_COUNT