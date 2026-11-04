
"""
SPECTRIX - 7-Day Wireless Channel Occupancy Database
Part A: Predictive Rule-Based Spectrum Allocation

Generates synthetic occupancy data for:
- 8 wireless channels
- 7 complete days
- 24 hours per day
- 10-second sampling interval

Output: data/raw/occupancy_7days.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

RANDOM_SEED = 42

DAYS = 7
SLOT_DURATION = 10  # seconds
START_DATE = "2026-11-02"

PERSISTENCE = 0.90

TOTAL_SLOTS_PER_DAY = 24 * 60 * 60 // SLOT_DURATION

EXPECTED_RECORDS = DAYS * TOTAL_SLOTS_PER_DAY * 8

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================================
# 2. CHANNEL CONFIGURATION
# ==========================================================

CHANNELS = [
    {"id": "CH01", "type": "Broadcast", "licensed": True, "offset": 0.02},
    {"id": "CH02", "type": "Broadcast", "licensed": True, "offset": -0.02},

    {"id": "CH03", "type": "Cellular", "licensed": True, "offset": -0.03},
    {"id": "CH04", "type": "Cellular", "licensed": True, "offset": 0.03},

    {"id": "CH05", "type": "Business", "licensed": True, "offset": 0.03},
    {"id": "CH06", "type": "Business", "licensed": True, "offset": -0.03},

    {"id": "CH07", "type": "ISM", "licensed": False, "offset": -0.03},
    {"id": "CH08", "type": "ISM", "licensed": False, "offset": 0.03},
]


# ==========================================================
# 3. DAILY OCCUPANCY PROFILES
# ==========================================================

# Values represent the target probability of a channel
# being busy during each hour (00:00 to 23:00).

CELLULAR_PROFILE = [
    0.12, 0.09, 0.08, 0.08, 0.10, 0.13,
    0.20, 0.30, 0.42, 0.50, 0.55, 0.58,
    0.62, 0.60, 0.58, 0.60, 0.65, 0.72,
    0.78, 0.82, 0.85, 0.75, 0.50, 0.25
]

BUSINESS_PROFILE = [
    0.03, 0.02, 0.02, 0.02, 0.02, 0.03,
    0.05, 0.12, 0.35, 0.70, 0.76, 0.78,
    0.78, 0.78, 0.77, 0.76, 0.72, 0.55,
    0.20, 0.08, 0.05, 0.04, 0.03, 0.03
]

ISM_PROFILE = [
    0.28, 0.27, 0.26, 0.25, 0.25, 0.27,
    0.30, 0.33, 0.35, 0.37, 0.38, 0.40,
    0.42, 0.41, 0.39, 0.38, 0.40, 0.43,
    0.45, 0.44, 0.41, 0.38, 0.34, 0.30
]


# ==========================================================
# 4. TIME PERIOD CLASSIFICATION
# ==========================================================

def classify_time_period(hour):

    if 0 <= hour < 5:
        return "Midnight"

    elif 5 <= hour < 8:
        return "Early Morning"

    elif 8 <= hour < 12:
        return "Morning"

    elif 12 <= hour < 17:
        return "Afternoon"

    elif 17 <= hour < 22:
        return "Evening"

    else:
        return "Late Night"


# ==========================================================
# 5. DUTY CYCLE FUNCTION
# ==========================================================

def get_duty_cycle(channel_type, hour, is_weekend, offset):

    if channel_type == "Broadcast":
        base_probability = 0.95

    elif channel_type == "Cellular":
        base_probability = CELLULAR_PROFILE[hour]

    elif channel_type == "Business":

        if is_weekend:
            base_probability = 0.02
        else:
            base_probability = BUSINESS_PROFILE[hour]

    elif channel_type == "ISM":
        base_probability = ISM_PROFILE[hour]

    else:
        raise ValueError("Unknown channel type")

    probability = base_probability + offset

    return float(np.clip(probability, 0.01, 0.99))


# ==========================================================
# 6. CHANNEL OCCUPANCY SIMULATION
# ==========================================================

def simulate_channel(channel, timestamps):

    channel_type = channel["type"]
    channel_id = channel["id"]
    offset = channel["offset"]

    total_slots = len(timestamps)

    busy_states = np.zeros(total_slots, dtype=np.int8)
    target_probabilities = np.zeros(total_slots)

    # Initialize channel state
    first_hour = timestamps[0].hour

    first_probability = get_duty_cycle(
        channel_type,
        first_hour,
        timestamps[0].weekday() >= 5,
        offset
    )

    state = int(rng.random() < first_probability)

    for i, timestamp in enumerate(timestamps):

        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5

        # Calculate expected busy probability
        target_probability = get_duty_cycle(
            channel_type,
            hour,
            is_weekend,
            offset
        )

        target_probabilities[i] = target_probability

        # Markov transition probabilities
        idle_to_busy = target_probability * (1 - PERSISTENCE)

        busy_to_idle = (1 - target_probability) * (1 - PERSISTENCE)

        random_value = rng.random()

        if state == 0:

            if random_value < idle_to_busy:
                state = 1

        else:

            if random_value < busy_to_idle:
                state = 0

        busy_states[i] = state

    # Create channel dataframe
    channel_data = pd.DataFrame({

        "timestamp": timestamps,

        "day_number": np.arange(total_slots) // TOTAL_SLOTS_PER_DAY + 1,

        "weekday": timestamps.day_name(),

        "hour": timestamps.hour,

        "time_period": [
            classify_time_period(hour)
            for hour in timestamps.hour
        ],

        "is_weekend": timestamps.weekday >= 5,

        "channel_id": channel_id,

        "channel_type": channel_type,

        "licensed": channel["licensed"],

        "target_busy_probability": target_probabilities,

        "busy_state": busy_states

    })

    return channel_data


# ==========================================================
# 7. GENERATE COMPLETE DATABASE
# ==========================================================

def generate_database():

    print("\n==========================================")
    print(" SPECTRIX CHANNEL OCCUPANCY GENERATOR")
    print("==========================================")

    print("\nGenerating timestamps...")

    total_slots = DAYS * TOTAL_SLOTS_PER_DAY

    timestamps = pd.date_range(
        start=START_DATE,
        periods=total_slots,
        freq=f"{SLOT_DURATION}s"
    )

    all_channel_data = []

    print("\nSimulating wireless channels...\n")

    for channel in CHANNELS:

        print(f"Generating {channel['id']} - {channel['type']}")

        channel_data = simulate_channel(
            channel,
            timestamps
        )

        all_channel_data.append(channel_data)

    # Combine all channel records
    database = pd.concat(
        all_channel_data,
        ignore_index=True
    )

    # Sort records by time and channel
    database.sort_values(
        by=["timestamp", "channel_id"],
        inplace=True
    )

    database.reset_index(drop=True, inplace=True)

    # Save CSV
    output_file = OUTPUT_DIR / "occupancy_7days.csv"

    database.to_csv(
        output_file,
        index=False
    )

    print("\n==========================================")
    print(" DATABASE GENERATION COMPLETED")
    print("==========================================")

    print(f"\nTotal records: {len(database):,}")

    print(f"Expected records: {EXPECTED_RECORDS:,}")

    print(f"Number of channels: {len(CHANNELS)}")

    print(f"Duration: {DAYS} days")

    print(f"Sampling interval: {SLOT_DURATION} seconds")

    print(f"\nSaved file: {output_file}")

    return database


# ==========================================================
# 8. DATABASE VALIDATION
# ==========================================================

def validate_database(database):

    print("\n==========================================")
    print(" DATABASE VALIDATION")
    print("==========================================")

    checks = {

        "Correct number of records":
            len(database) == EXPECTED_RECORDS,

        "Exactly 8 channels":
            database["channel_id"].nunique() == 8,

        "Exactly 7 days":
            database["day_number"].nunique() == 7,

        "24-hour coverage":
            database["hour"].nunique() == 24,

        "Only binary occupancy values":
            set(database["busy_state"].unique()).issubset({0, 1}),

        "No missing values":
            not database.isnull().any().any(),

        "No duplicate records":
            not database.duplicated(
                subset=["timestamp", "channel_id"]
            ).any()

    }

    for check_name, result in checks.items():

        status = "PASS" if result else "FAIL"

        print(f"{check_name}: {status}")

    if not all(checks.values()):
        raise ValueError("Database validation failed!")

    print("\nAll validation checks passed!")


# ==========================================================
# 9. BASIC OCCUPANCY REPORT
# ==========================================================

def generate_basic_report(database):

    print("\n==========================================")
    print(" CHANNEL OCCUPANCY SUMMARY")
    print("==========================================")

    summary = database.groupby(
        ["channel_id", "channel_type"]
    )["busy_state"].mean() * 100

    print("\nAverage busy percentage by channel:\n")

    print(summary.round(2).to_string())

    print("\nOccupancy by time period:\n")

    period_summary = database.groupby(
        "time_period",
        observed=False
    )["busy_state"].mean() * 100

    print(period_summary.round(2).to_string())

    print("\nOccupancy by weekday:\n")

    weekday_summary = database.groupby(
        "weekday"
    )["busy_state"].mean() * 100

    print(weekday_summary.round(2).to_string())


# ==========================================================
# 10. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    df = generate_database()

    validate_database(df)

    generate_basic_report(df)

    print("\nSPECTRIX DATABASE MODULE FINISHED SUCCESSFULLY!")
