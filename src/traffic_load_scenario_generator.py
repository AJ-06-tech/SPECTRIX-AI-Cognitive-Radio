
"""
SPECTRIX - Traffic Load Scenario Generator

Generates reproducible low, normal and high traffic scenarios.
Existing project files and experiment results are not modified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from channel_quality import ChannelQualityModel
from secondary_user import SecondaryUserEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_ROOT = (
    BASE_DIR
    / "data"
    / "processed"
    / "traffic_load_testing"
)

ORIGINAL_SCENARIO = (
    BASE_DIR
    / "data"
    / "processed"
    / "baseline_comparison"
    / "common_scenarios_7days.csv"
)

TRAINING_END = pd.Timestamp("2026-11-07 00:00:00")

SLOT_DURATION_SECONDS = 10

LOAD_CONFIGURATIONS = {
    "low": 0.5,
    "normal": 1.0,
    "high": 1.5,
}


# ============================================================
# TIME PERIOD CLASSIFICATION
# ============================================================

def get_time_period(timestamp):

    hour = pd.Timestamp(timestamp).hour

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

    return "Late Night"


# ============================================================
# JSON CONVERSION
# ============================================================

def convert_to_json(value):

    if isinstance(value, dict):

        return {
            str(key): convert_to_json(item)
            for key, item in value.items()
        }

    elif isinstance(value, (list, tuple)):

        return [
            convert_to_json(item)
            for item in value
        ]

    elif isinstance(value, np.generic):

        return value.item()

    elif isinstance(value, pd.Timestamp):

        return str(value)

    return value


def to_json(value):

    return json.dumps(
        convert_to_json(value),
        default=str
    )


# ============================================================
# APPLY TRAFFIC MULTIPLIER
# ============================================================

def configure_traffic_load(user_environment, multiplier):

    for user in user_environment.users.values():

        original_probabilities = user.REQUEST_PROBABILITIES

        user.REQUEST_PROBABILITIES = {
            period: min(probability * multiplier, 1.0)
            for period, probability in original_probabilities.items()
        }


# ============================================================
# NORMAL LOAD VALIDATION
# ============================================================

def validate_normal_load(scenarios_df):

    print("\nVALIDATING NORMAL LOAD AGAINST ORIGINAL SCENARIOS")

    if not ORIGINAL_SCENARIO.exists():

        raise FileNotFoundError(
            f"Original scenario file not found: {ORIGINAL_SCENARIO}"
        )

    original_df = pd.read_csv(ORIGINAL_SCENARIO)

    if len(original_df) != len(scenarios_df):

        raise ValueError(
            "Normal-load slot count differs from original scenarios."
        )

    columns_to_compare = [
        "simulation_time",
        "time_period",
        "requests_generated",
        "requesting_users",
        "generated_requests",
        "channel_conditions",
        "sensing_reports",
        "cooperative_decisions",
        "actual_channel_status",
    ]

    for column in columns_to_compare:

        if not original_df[column].equals(scenarios_df[column]):

            raise ValueError(
                f"Normal-load validation failed for column: {column}"
            )

    print("Normal-load validation: PASSED")
    print("All selected scenario inputs match the original dataset.")


# ============================================================
# GENERATE SCENARIO
# ============================================================

def generate_scenario(load_name):

    multiplier = LOAD_CONFIGURATIONS[load_name]

    print("\n" + "=" * 75)
    print(f"SPECTRIX - {load_name.upper()} TRAFFIC SCENARIO GENERATION")
    print("=" * 75)

    print(f"Traffic multiplier: {multiplier}x")

    # --------------------------------------------------------
    # STEP 1: INITIALIZE MODULES
    # Keep the same order as the original scenario generator.
    # --------------------------------------------------------

    print("\nSTEP 1: INITIALIZING MODULES")

    channel_simulator = ChannelSimulator()

    sensing_engine = SpectrumSensing()

    quality_model = ChannelQualityModel()

    user_environment = SecondaryUserEnvironment()

    # Change only the request probabilities.
    configure_traffic_load(
        user_environment,
        multiplier
    )

    timestamps = channel_simulator.get_all_timestamps()

    if len(timestamps) != 60480:

        raise ValueError(
            f"Expected 60480 slots, found {len(timestamps)}"
        )

    print("Total simulation slots:", len(timestamps))
    print("Secondary users:", len(user_environment.users))

    # --------------------------------------------------------
    # STEP 2: GENERATE SCENARIOS
    # --------------------------------------------------------

    print("\nSTEP 2: GENERATING TRAFFIC SCENARIOS")

    scenario_records = []

    total_requests = 0

    for slot_index, simulation_time in enumerate(timestamps):

        simulation_time = pd.Timestamp(simulation_time)

        time_period = get_time_period(simulation_time)

        # A. Actual channel occupancy

        channel_status = (
            channel_simulator.get_channel_status(
                simulation_time
            )
        )

        # B. Channel quality

        channel_conditions = (
            quality_model.generate_conditions(
                channel_status
            )
        )

        # C. Individual sensing

        sensing_reports = (
            sensing_engine.perform_sensing(
                channel_status
            )
        )

        # D. Cooperative sensing

        cooperative_results = (
            sensing_engine.cooperative_sensing(
                sensing_reports
            )
        )

        sensing_decisions = {
            channel_id: result["decision"]
            for channel_id, result
            in cooperative_results.items()
        }

        # E. Traffic generation

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

        # F. Save scenario inputs

        scenario_records.append({

            "simulation_time": str(simulation_time),

            "time_period": time_period,

            "is_training_period": (
                simulation_time < TRAINING_END
            ),

            "requests_generated": len(
                generated_requests
            ),

            "requesting_users": to_json(
                requesting_users
            ),

            "generated_requests": to_json(
                generated_requests
            ),

            "channel_conditions": to_json(
                channel_conditions
            ),

            "sensing_reports": to_json(
                sensing_reports
            ),

            "cooperative_decisions": to_json(
                sensing_decisions
            ),

            "actual_channel_status": to_json(
                channel_status
            ),

        })

        # G. Progress

        if (slot_index + 1) % 8640 == 0:

            completed_days = (slot_index + 1) // 8640

            print(
                f"Day {completed_days}/7 completed | "
                f"Requests: {total_requests}"
            )

    # --------------------------------------------------------
    # STEP 3: SAVE DATASET
    # --------------------------------------------------------

    print("\nSTEP 3: SAVING SCENARIO DATASET")

    scenarios_df = pd.DataFrame(scenario_records)

    output_dir = OUTPUT_ROOT / f"{load_name}_load"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    scenario_path = (
        output_dir / "common_scenarios_7days.csv"
    )

    scenarios_df.to_csv(
        scenario_path,
        index=False
    )

    print("Dataset saved:", scenario_path)

    # --------------------------------------------------------
    # STEP 4: VALIDATE DATASET
    # --------------------------------------------------------

    print("\nSTEP 4: VALIDATING DATASET")

    validation_results = {

        "correct_slot_count":
            len(scenarios_df) == 60480,

        "no_missing_timestamps":
            scenarios_df["simulation_time"].notna().all(),

        "no_duplicate_timestamps":
            not scenarios_df[
                "simulation_time"
            ].duplicated().any(),

        "all_channel_status_present":
            scenarios_df[
                "actual_channel_status"
            ].notna().all(),

        "all_sensing_reports_present":
            scenarios_df[
                "sensing_reports"
            ].notna().all(),

        "all_channel_conditions_present":
            scenarios_df[
                "channel_conditions"
            ].notna().all(),

        "all_traffic_records_present":
            scenarios_df[
                "generated_requests"
            ].notna().all(),

    }

    for test_name, result in validation_results.items():

        print(
            f"{test_name}: "
            f"{'PASSED' if result else 'FAILED'}"
        )

    if not all(validation_results.values()):

        raise ValueError(
            "Scenario validation failed."
        )

    # --------------------------------------------------------
    # STEP 5: NORMAL LOAD COMPARISON
    # --------------------------------------------------------

    if load_name == "normal":

        validate_normal_load(scenarios_df)

    # --------------------------------------------------------
    # STEP 6: SAVE METADATA
    # --------------------------------------------------------

    training_slots = int(
        scenarios_df["is_training_period"].sum()
    )

    testing_slots = (
        len(scenarios_df) - training_slots
    )

    metadata = {

        "project": "SPECTRIX",

        "experiment": "Traffic Load Robustness",

        "traffic_load": load_name,

        "traffic_multiplier": multiplier,

        "simulation_days": 7,

        "slot_duration_seconds": SLOT_DURATION_SECONDS,

        "total_slots": len(scenarios_df),

        "total_channels": 8,

        "secondary_users": len(user_environment.users),

        "total_packet_requests": total_requests,

        "training_slots": training_slots,

        "held_out_test_slots": testing_slots,

        "training_period_end": str(TRAINING_END),

        "held_out_test_start": str(TRAINING_END),

        "held_out_test_end": str(
            pd.Timestamp(timestamps[-1])
        ),

        "random_seed": 42,

        "normal_load_matches_original": (
            load_name == "normal"
        ),

        "note": (
            "Synthetic simulation scenarios. "
            "Only secondary-user request probabilities "
            "are scaled between traffic configurations."
        ),

    }

    metadata_path = (
        output_dir / "scenario_metadata.json"
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

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print(f"{load_name.upper()} TRAFFIC SCENARIO COMPLETED")
    print("=" * 75)

    print("Traffic multiplier:", multiplier)
    print("Total slots:", len(scenarios_df))
    print("Total requests:", total_requests)
    print("Training slots:", training_slots)
    print("Held-out slots:", testing_slots)
    print("Output directory:", output_dir)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="SPECTRIX Traffic Load Scenario Generator"
    )

    parser.add_argument(
        "--load",
        required=True,
        choices=["low", "normal", "high"],
        help="Traffic load configuration"
    )

    args = parser.parse_args()

    generate_scenario(args.load)


if __name__ == "__main__":

    main()
