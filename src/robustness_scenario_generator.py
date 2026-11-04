
"""
SPECTRIX - Robustness Scenario Generator

Generates separate seven-day scenarios for 3, 5, 8 and 10 users.

Original scenario_generator.py and baseline comparison data
remain unchanged.
"""

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_DIR,
    CHANNEL_IDS,
    RANDOM_SEED,
    SLOT_DURATION_SECONDS
)

from channel_simulator import ChannelSimulator
from channel_quality import ChannelQualityModel

from robustness_config import (
    SCALABILITY_USER_COUNTS,
    get_experiment_config
)

from robustness_spectrum_sensing import (
    RobustnessSpectrumSensing
)

from robustness_secondary_user import (
    RobustnessSecondaryUserEnvironment
)


OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "robustness_testing"
)

TRAINING_END = pd.Timestamp("2026-11-07 00:00:00")


def get_time_period(timestamp):

    hour = pd.Timestamp(timestamp).hour

    if 0 <= hour < 5:
        return "Midnight"

    if 5 <= hour < 8:
        return "Early Morning"

    if 8 <= hour < 12:
        return "Morning"

    if 12 <= hour < 17:
        return "Afternoon"

    if 17 <= hour < 22:
        return "Evening"

    return "Late Night"


def convert_to_json(value):

    if isinstance(value, dict):
        return {
            str(key): convert_to_json(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            convert_to_json(item)
            for item in value
        ]

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, (pd.Timestamp,)):
        return str(value)

    return value


def to_json(value):

    return json.dumps(
        convert_to_json(value),
        default=str
    )


def generate_scenario(user_count):

    print("\n" + "=" * 75)
    print(f"GENERATING SCENARIOS FOR {user_count} USERS")
    print("=" * 75)

    settings = get_experiment_config(user_count)

    experiment_dir = OUTPUT_DIR / f"users_{user_count}"

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Initialize fresh instances for reproducible experiments.
    channel_simulator = ChannelSimulator()

    sensing_engine = RobustnessSpectrumSensing(
        user_count=user_count,
        verbose=False
    )

    quality_model = ChannelQualityModel()

    user_environment = RobustnessSecondaryUserEnvironment(
        user_count=user_count,
        verbose=False
    )

    timestamps = channel_simulator.get_all_timestamps()

    if len(timestamps) != 60480:
        raise ValueError(
            f"Expected 60480 slots, found {len(timestamps)}"
        )

    scenario_records = []

    total_requests = 0

    for slot_index, simulation_time in enumerate(timestamps):

        simulation_time = pd.Timestamp(simulation_time)

        time_period = get_time_period(simulation_time)

        # 1. Actual channel occupancy
        channel_status = (
            channel_simulator.get_channel_status(
                simulation_time
            )
        )

        # 2. Channel quality
        channel_conditions = (
            quality_model.generate_conditions(
                channel_status
            )
        )

        # 3. Individual sensing
        sensing_reports = (
            sensing_engine.perform_sensing(
                channel_status
            )
        )

        # 4. Cooperative sensing
        cooperative_results = (
            sensing_engine.cooperative_sensing(
                sensing_reports
            )
        )

        sensing_decisions = {
            channel_id: result["decision"]
            for channel_id, result in cooperative_results.items()
        }

        # 5. Traffic generation
        generated_requests = (
            user_environment.generate_requests(
                time_period,
                simulation_time
            )
        )

        requesting_users = (
            user_environment.get_requesting_users()
        )

        total_requests += len(generated_requests)

        # 6. Store common scenario inputs
        scenario_records.append({

            "simulation_time": str(simulation_time),

            "time_period": time_period,

            "is_training_period": (
                simulation_time < TRAINING_END
            ),

            "requests_generated": len(generated_requests),

            "requesting_users": to_json(requesting_users),

            "generated_requests": to_json(generated_requests),

            "channel_conditions": to_json(channel_conditions),

            "sensing_reports": to_json(sensing_reports),

            "cooperative_decisions": to_json(sensing_decisions),

            # Kept separately for transmission outcome evaluation.
            # Never pass this field to either allocator.
            "actual_channel_status": to_json(channel_status)
        })

        if (slot_index + 1) % 8640 == 0:

            completed_days = (slot_index + 1) // 8640

            print(
                f"Day {completed_days}/7 completed | "
                f"Requests: {total_requests}"
            )

    # Save scenario dataset
    scenarios_df = pd.DataFrame(scenario_records)

    scenario_path = (
        experiment_dir / "common_scenarios.csv"
    )

    scenarios_df.to_csv(
        scenario_path,
        index=False
    )

    # Metadata
    training_slots = int(
        scenarios_df["is_training_period"].sum()
    )

    testing_slots = (
        len(scenarios_df) - training_slots
    )

    metadata = {

        "project": "SPECTRIX",

        "experiment": "Secondary user scalability",

        "dataset_type": "Synthetic simulation scenarios",

        "secondary_user_count": user_count,

        "secondary_user_ids": settings["secondary_user_ids"],

        "simulation_days": 7,

        "slot_duration_seconds": SLOT_DURATION_SECONDS,

        "total_slots": len(scenarios_df),

        "total_channels": len(CHANNEL_IDS),

        "total_packet_requests": total_requests,

        "individual_detection_probability":
            settings["individual_detection_probability"],

        "individual_false_alarm_probability":
            settings["individual_false_alarm_probability"],

        "required_busy_votes":
            settings["required_busy_votes"],

        "cooperative_detection_probability":
            settings["cooperative_detection_probability"],

        "cooperative_false_alarm_probability":
            settings["cooperative_false_alarm_probability"],

        "hmm_training_days": 5,

        "hmm_testing_days": 2,

        "hmm_training_slots": training_slots,

        "held_out_test_slots": testing_slots,

        "training_period_end_exclusive": str(TRAINING_END),

        "held_out_test_start": str(TRAINING_END),

        "held_out_test_end": str(
            pd.Timestamp(timestamps[-1])
        ),

        "random_seed": RANDOM_SEED,

        "user_random_seed_rule": (
            "Each user uses RANDOM_SEED + 1000 + numeric user ID"
        ),

        "sensing_random_seed": RANDOM_SEED + 2,

        "channel_quality_random_seed": RANDOM_SEED + 5,

        "channel_occupancy_source": (
            "Original seven-day synthetic occupancy dataset"
        ),

        "same_scenarios_for_both_algorithms": True
    }

    metadata_path = (
        experiment_dir / "scenario_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # Validation
    validation = {

        "correct_slot_count":
            len(scenarios_df) == 60480,

        "correct_user_count":
            all(
                len(json.loads(value)) <= user_count
                for value in scenarios_df["requesting_users"]
            ),

        "no_missing_timestamps":
            scenarios_df["simulation_time"].notna().all(),

        "no_duplicate_timestamps":
            not scenarios_df["simulation_time"].duplicated().any(),

        "all_sensing_reports_present":
            scenarios_df["sensing_reports"].notna().all(),

        "all_channel_conditions_present":
            scenarios_df["channel_conditions"].notna().all(),

        "all_traffic_records_present":
            scenarios_df["generated_requests"].notna().all(),

        "correct_training_slots":
            training_slots == 43200,

        "correct_testing_slots":
            testing_slots == 17280
    }

    print("\nVALIDATION RESULTS")

    for name, passed in validation.items():

        print(
            f"{name}: {'PASSED' if passed else 'FAILED'}"
        )

    if not all(validation.values()):
        raise ValueError(
            f"Scenario validation failed for {user_count} users."
        )

    print("\nSCENARIO GENERATION SUCCESSFUL")

    print("Users:", user_count)

    print("Total slots:", len(scenarios_df))

    print("Total requests:", total_requests)

    print("Training slots:", training_slots)

    print("Testing slots:", testing_slots)

    print("Saved:", scenario_path)

    # Release memory before next experiment.
    del scenarios_df
    del scenario_records

    gc.collect()

    return total_requests


def main():

    print("\n" + "=" * 75)
    print("SPECTRIX - USER SCALABILITY SCENARIO GENERATION")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = []

    for user_count in SCALABILITY_USER_COUNTS:

        total_requests = generate_scenario(user_count)

        summary.append({
            "users": user_count,
            "total_requests": total_requests
        })

    summary_df = pd.DataFrame(summary)

    summary_path = (
        OUTPUT_DIR / "scenario_generation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 75)
    print("ALL SCALABILITY SCENARIOS GENERATED")
    print("=" * 75)

    print(summary_df.to_string(index=False))

    print("\nSummary saved:", summary_path)


if __name__ == "__main__":

    main()
