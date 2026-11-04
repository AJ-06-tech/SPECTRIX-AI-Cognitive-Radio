
import json
import numpy as np
import pandas as pd

from pathlib import Path

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from channel_quality import ChannelQualityModel
from secondary_user import SecondaryUserEnvironment


# ============================================================
# SPECTRIX - COMMON SCENARIO GENERATOR
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "baseline_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOT_DURATION_SECONDS = 10

TRAINING_END = pd.Timestamp("2026-11-07 00:00:00")


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

    elif isinstance(value, (pd.Timestamp,)):

        return str(value)

    return value


def to_json(value):

    return json.dumps(
        convert_to_json(value),
        default=str
    )


# ============================================================
# MAIN SCENARIO GENERATOR
# ============================================================

def main():

    print("\n" + "=" * 75)

    print(" SPECTRIX - COMMON SCENARIO GENERATION")

    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1: INITIALIZE MODULES
    # --------------------------------------------------------

    print("\nSTEP 1: INITIALIZING SIMULATION MODULES")

    channel_simulator = ChannelSimulator()

    sensing_engine = SpectrumSensing()

    quality_model = ChannelQualityModel()

    user_environment = SecondaryUserEnvironment()

    timestamps = channel_simulator.get_all_timestamps()

    if len(timestamps) != 60480:

        raise ValueError(
            f"Expected 60480 slots, found {len(timestamps)}"
        )

    print("\nAll modules initialized.")

    print("Total simulation slots:", len(timestamps))

    # --------------------------------------------------------
    # STEP 2: GENERATE COMMON SCENARIOS
    # --------------------------------------------------------

    print("\nSTEP 2: GENERATING COMMON SCENARIOS")

    scenario_records = []

    total_requests = 0

    for slot_index, simulation_time in enumerate(timestamps):

        simulation_time = pd.Timestamp(simulation_time)

        time_period = get_time_period(simulation_time)

        # ----------------------------------------------------
        # A. ACTUAL CHANNEL OCCUPANCY
        # ----------------------------------------------------

        channel_status = (
            channel_simulator.get_channel_status(
                simulation_time
            )
        )

        # ----------------------------------------------------
        # B. CHANNEL QUALITY
        # ----------------------------------------------------

        channel_conditions = (
            quality_model.generate_conditions(
                channel_status
            )
        )

        # ----------------------------------------------------
        # C. INDIVIDUAL SENSING REPORTS
        # ----------------------------------------------------

        sensing_reports = (
            sensing_engine.perform_sensing(
                channel_status
            )
        )

        # ----------------------------------------------------
        # D. COOPERATIVE SENSING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # E. SECONDARY USER TRAFFIC
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # F. SAVE SCENARIO INPUTS
        # ----------------------------------------------------

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

            # Ground truth is stored separately.
            # It must NOT be exposed to either allocator.
            "actual_channel_status": to_json(
                channel_status
            )

        })

        # ----------------------------------------------------
        # G. DAILY PROGRESS
        # ----------------------------------------------------

        if (slot_index + 1) % 8640 == 0:

            completed_days = (
                slot_index + 1
            ) // 8640

            print(

                f"Day {completed_days}/7 completed | "

                f"Requests generated: {total_requests}"

            )

    # --------------------------------------------------------
    # STEP 3: SAVE COMMON SCENARIO DATASET
    # --------------------------------------------------------

    print("\nSTEP 3: SAVING COMMON SCENARIO DATASET")

    scenarios_df = pd.DataFrame(scenario_records)

    scenario_path = (
        OUTPUT_DIR / "common_scenarios_7days.csv"
    )

    scenarios_df.to_csv(
        scenario_path,
        index=False
    )

    print("\nScenario dataset saved.")

    print(scenario_path)

    # --------------------------------------------------------
    # STEP 4: SAVE SCENARIO METADATA
    # --------------------------------------------------------

    training_slots = int(
        scenarios_df["is_training_period"].sum()
    )

    testing_slots = (
        len(scenarios_df) - training_slots
    )

    metadata = {

        "project": "SPECTRIX",

        "dataset_type": "Synthetic simulation scenarios",

        "simulation_days": 7,

        "slot_duration_seconds": SLOT_DURATION_SECONDS,

        "total_slots": len(scenarios_df),

        "total_channels": 8,

        "secondary_users": 5,

        "total_packet_requests": total_requests,

        "training_warmup_slots": training_slots,

        "held_out_test_slots": testing_slots,

        "training_period_end": str(TRAINING_END),

        "held_out_test_start": str(TRAINING_END),

        "held_out_test_end": str(
            pd.Timestamp(timestamps[-1])
        ),

        "primary_comparison_window": (
            "2026-11-07 to 2026-11-08"
        ),

        "random_seed_note": (
            "Uses the existing module-level random seeds. "
            "Saved scenarios are intended to be replayed "
            "by both algorithms."
        )

    }

    metadata_path = (
        OUTPUT_DIR / "scenario_metadata.json"
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
    # STEP 5: VALIDATE GENERATED DATASET
    # --------------------------------------------------------

    print("\nSTEP 4: VALIDATING SCENARIOS")

    validation_results = {

        "correct_slot_count":
            len(scenarios_df) == 60480,

        "no_missing_timestamps":
            scenarios_df["simulation_time"].notna().all(),

        "no_duplicate_timestamps":
            not scenarios_df[
                "simulation_time"
            ].duplicated().any(),

        "all_channel_status_records_present":
            scenarios_df[
                "actual_channel_status"
            ].notna().all(),

        "all_sensing_reports_present":
            scenarios_df[
                "sensing_reports"
            ].notna().all(),

        "all_quality_records_present":
            scenarios_df[
                "channel_conditions"
            ].notna().all(),

        "all_traffic_records_present":
            scenarios_df[
                "generated_requests"
            ].notna().all()

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
    # STEP 6: DISPLAY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" COMMON SCENARIO GENERATION COMPLETED")

    print("=" * 75)

    print("\nTotal slots:", len(scenarios_df))

    print("Total requests:", total_requests)

    print("Training/warm-up slots:", training_slots)

    print("Held-out test slots:", testing_slots)

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nGenerated files:")

    print("1. common_scenarios_7days.csv")

    print("2. scenario_metadata.json")

    print("\nBoth algorithms will use this saved scenario dataset.")

    print("\n" + "=" * 75)


if __name__ == "__main__":

    main()
