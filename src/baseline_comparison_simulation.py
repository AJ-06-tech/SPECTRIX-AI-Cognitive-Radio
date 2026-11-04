
import copy
import hashlib
import json

import numpy as np
import pandas as pd

from pathlib import Path

try:
    from .config import (
        SECONDARY_USER_IDS,
        MAX_TRANSMISSION_ATTEMPTS,
        PACKET_SIZE_BYTES
    )

    from .hmm_predictor import HMMPredictor
    from .assm_allocator import ASSMAllocator
    from .baseline_allocator import BaselineAllocator

except ImportError:

    from config import (
        SECONDARY_USER_IDS,
        MAX_TRANSMISSION_ATTEMPTS,
        PACKET_SIZE_BYTES
    )

    from hmm_predictor import HMMPredictor
    from assm_allocator import ASSMAllocator
    from baseline_allocator import BaselineAllocator


# ============================================================
# SPECTRIX - CONTROLLED BASELINE VS ASSM COMPARISON
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SCENARIO_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "baseline_comparison"
    / "common_scenarios_7days.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
    / "baseline_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_START = pd.Timestamp("2026-11-07 00:00:00")

TEST_END = pd.Timestamp("2026-11-09 00:00:00")

EXPECTED_TEST_SLOTS = 17280

SLOT_DURATION_SECONDS = 10

ALGORITHMS = [
    "BASELINE",
    "ASSM"
]


# ============================================================
# JSON HELPERS
# ============================================================

def parse_json(value):

    if isinstance(value, str):

        return json.loads(value)

    return value


# ============================================================
# STABLE PACKET IDENTIFIER
# ============================================================

def create_packet_id(
    user_id,
    created_at,
    request_position
):

    """
    Creates the same ID for the same original packet
    in both algorithms.

    It does not depend on queue length, delivery order,
    or the algorithm's current number of requests.
    """

    timestamp = pd.Timestamp(
        created_at
    ).isoformat()

    return (
        f"{user_id}|{timestamp}|{request_position}"
    )


# ============================================================
# DETERMINISTIC TRANSMISSION RANDOMNESS
# ============================================================

def common_random_value(
    user_id,
    packet_id,
    attempt_number
):

    """
    Generates a repeatable random draw.

    The same packet and attempt receive the same
    random draw in BASELINE and ASSM.
    """

    key = (

        f"{user_id}|{packet_id}|{attempt_number}"

    ).encode("utf-8")

    digest = hashlib.sha256(
        key
    ).digest()

    integer_value = int.from_bytes(
        digest[:8],
        byteorder="big"
    )

    return integer_value / (2 ** 64)


# ============================================================
# TRANSMISSION MODEL
# ============================================================

def simulate_transmission(
    user_id,
    request,
    channel_id,
    channel_status,
    channel_snr
):

    request["attempts"] += 1

    attempt_number = request["attempts"]

    random_value = common_random_value(

        user_id,

        request["packet_id"],

        attempt_number

    )

    # --------------------------------------------------------
    # ACTUAL CHANNEL OCCUPANCY
    # --------------------------------------------------------

    # Ground truth is used only after channel selection.
    # It is never exposed to either allocation algorithm.

    actual_status = channel_status[
        channel_id
    ]["status"]

    # Busy licensed channel or external ISM activity.
    if actual_status != "AVAILABLE":

        return {

            "status": "COLLISION",

            "attempt_number": attempt_number,

            "packet_failed": (
                attempt_number >= MAX_TRANSMISSION_ATTEMPTS
            ),

            "bytes_transmitted": 0

        }

    # --------------------------------------------------------
    # TRANSMISSION SUCCESS PROBABILITY
    # --------------------------------------------------------

    success_probability = float(

        np.clip(

            0.50 + channel_snr / 60.0,

            0.10,

            0.98

        )

    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    if random_value < success_probability:

        return {

            "status": "SUCCESS",

            "attempt_number": attempt_number,

            "packet_failed": False,

            "bytes_transmitted": request.get(

                "packet_size",

                PACKET_SIZE_BYTES

            )

        }

    # --------------------------------------------------------
    # FAILED RF ATTEMPT
    # --------------------------------------------------------

    return {

        "status": (

            "FAILED"

            if attempt_number >= MAX_TRANSMISSION_ATTEMPTS

            else "RETRY_REQUIRED"

        ),

        "attempt_number": attempt_number,

        "packet_failed": (

            attempt_number >= MAX_TRANSMISSION_ATTEMPTS

        ),

        "bytes_transmitted": 0

    }


# ============================================================
# INITIALIZE ALGORITHM STATE
# ============================================================

def initialize_algorithm_state():

    states = {}

    for algorithm in ALGORITHMS:

        states[algorithm] = {

            "queues": {

                user_id: []

                for user_id in SECONDARY_USER_IDS

            },

            "requests": 0,

            "delivered": 0,

            "failed_packets": 0,

            "attempts": 0,

            "successful_attempts": 0,

            "failed_attempts": 0,

            "collisions": 0,

            "handovers": 0,

            "bytes_transmitted": 0,

            "total_latency": 0.0,

            "completed_packets": 0,

            "allocated_transmissions": 0,

            "user_delivered": {

                user_id: 0

                for user_id in SECONDARY_USER_IDS

            },

            "user_latency": {

                user_id: []

                for user_id in SECONDARY_USER_IDS

            },

            "channel_allocations": {},

            "slot_records": []

        }

    return states


# ============================================================
# ADD COMMON REQUESTS TO ALGORITHM QUEUES
# ============================================================

def add_generated_requests(
    state,
    generated_requests
):

    """
    Adds identical original requests to the separate
    queues of each algorithm.

    Packet IDs are independent of queue state.
    """

    new_requests = 0

    for request_position, original_request in enumerate(
        generated_requests
    ):

        request = copy.deepcopy(
            original_request
        )

        user_id = request["user_id"]

        if user_id not in state["queues"]:

            raise ValueError(
                f"Unknown secondary user: {user_id}"
            )

        request["attempts"] = 0

        request["last_channel"] = None

        request["packet_id"] = create_packet_id(

            user_id,

            request["created_at"],

            request_position

        )

        state["queues"][user_id].append(
            request
        )

        new_requests += 1

    state["requests"] += new_requests

    return new_requests


# ============================================================
# PROCESS ONE ALGORITHM FOR ONE SLOT
# ============================================================

def process_algorithm_slot(
    algorithm,
    state,
    scenario,
    allocator
):

    simulation_time = pd.Timestamp(
        scenario["simulation_time"]
    )

    channel_status = parse_json(
        scenario["actual_channel_status"]
    )

    channel_conditions = parse_json(
        scenario["channel_conditions"]
    )

    sensing_decisions = parse_json(
        scenario["cooperative_decisions"]
    )

    generated_requests = parse_json(
        scenario["generated_requests"]
    )

    # --------------------------------------------------------
    # 1. ADD NEW PACKETS
    # --------------------------------------------------------

    new_requests = add_generated_requests(

        state,

        generated_requests

    )

    # --------------------------------------------------------
    # 2. FIND USERS WITH PENDING PACKETS
    # --------------------------------------------------------

    requesting_users = [

        user_id

        for user_id in SECONDARY_USER_IDS

        if state["queues"][user_id]

    ]

    # --------------------------------------------------------
    # 3. CHANNEL ALLOCATION
    # --------------------------------------------------------

    if requesting_users:

        if algorithm == "BASELINE":

            allocations = allocator.allocate_channels(

                requesting_users,

                sensing_decisions

            )

        else:

            allocations = allocator.allocate_channels(

                requesting_users,

                sensing_decisions,

                state["hmm_forecasts"],

                channel_conditions

            )

    else:

        allocations = {}

    # --------------------------------------------------------
    # 4. INITIALIZE SLOT METRICS
    # --------------------------------------------------------

    successful_this_slot = 0

    collisions_this_slot = 0

    attempts_this_slot = 0

    bytes_this_slot = 0

    allocated_this_slot = 0

    # --------------------------------------------------------
    # 5. TRANSMIT ALLOCATED PACKETS
    # --------------------------------------------------------

    for user_id, allocation in allocations.items():

        if allocation["status"] != "ALLOCATED":

            continue

        channel_id = allocation["channel_id"]

        queue = state["queues"][user_id]

        if not queue:

            continue

        request = queue[0]

        previous_channel = request["last_channel"]

        # ----------------------------------------------------
        # HANDOVER
        # ----------------------------------------------------

        if (

            request["attempts"] > 0

            and previous_channel is not None

            and previous_channel != channel_id

        ):

            state["handovers"] += 1

        request["last_channel"] = channel_id

        # ----------------------------------------------------
        # TRANSMISSION
        # ----------------------------------------------------

        channel_snr = channel_conditions[
            channel_id
        ]["snr"]

        result = simulate_transmission(

            user_id,

            request,

            channel_id,

            channel_status,

            channel_snr

        )

        attempts_this_slot += 1

        state["attempts"] += 1

        state["allocated_transmissions"] += 1

        allocated_this_slot += 1

        state["channel_allocations"][channel_id] = (

            state["channel_allocations"].get(

                channel_id,

                0

            ) + 1

        )

        # ----------------------------------------------------
        # SUCCESSFUL TRANSMISSION
        # ----------------------------------------------------

        if result["status"] == "SUCCESS":

            successful_this_slot += 1

            state["delivered"] += 1

            state["successful_attempts"] += 1

            packet_bytes = result["bytes_transmitted"]

            bytes_this_slot += packet_bytes

            state["bytes_transmitted"] += packet_bytes

            created_at = pd.Timestamp(
                request["created_at"]
            )

            completion_time = (

                simulation_time

                + pd.Timedelta(
                    seconds=SLOT_DURATION_SECONDS
                )

            )

            latency = (

                completion_time - created_at

            ).total_seconds()

            state["total_latency"] += latency

            state["completed_packets"] += 1

            state["user_delivered"][user_id] += 1

            state["user_latency"][user_id].append(
                latency
            )

            queue.pop(0)

        # ----------------------------------------------------
        # COLLISION
        # ----------------------------------------------------

        elif result["status"] == "COLLISION":

            collisions_this_slot += 1

            state["collisions"] += 1

            state["failed_attempts"] += 1

            if result["packet_failed"]:

                state["failed_packets"] += 1

                queue.pop(0)

        # ----------------------------------------------------
        # OTHER FAILED ATTEMPT
        # ----------------------------------------------------

        else:

            state["failed_attempts"] += 1

            if result["packet_failed"]:

                state["failed_packets"] += 1

                queue.pop(0)

    # --------------------------------------------------------
    # 6. SAVE SLOT RESULTS
    # --------------------------------------------------------

    state["slot_records"].append({

        "simulation_time": str(simulation_time),

        "time_period": scenario["time_period"],

        "requests_generated": new_requests,

        "requesting_users": len(requesting_users),

        "allocated_users": allocated_this_slot,

        "transmission_attempts": attempts_this_slot,

        "successful_packets": successful_this_slot,

        "collisions": collisions_this_slot,

        "bytes_transmitted": bytes_this_slot

    })


# ============================================================
# JAIN'S FAIRNESS INDEX
# ============================================================

def calculate_jain_index(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:

        return 0.0

    denominator = (

        len(values) * np.sum(values ** 2)

    )

    if denominator == 0:

        return 0.0

    return float(

        np.sum(values) ** 2 / denominator

    )


# ============================================================
# SAVE RESULTS FOR ONE ALGORITHM
# ============================================================

def save_algorithm_results(
    algorithm,
    state,
    total_slots
):

    report_dir = OUTPUT_DIR / algorithm

    report_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SLOT-WISE RESULTS
    # --------------------------------------------------------

    slots_df = pd.DataFrame(
        state["slot_records"]
    )

    slots_df.to_csv(

        report_dir / "slot_results.csv",

        index=False

    )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    total_seconds = (

        total_slots * SLOT_DURATION_SECONDS

    )

    pending_packets = sum(

        len(queue)

        for queue in state["queues"].values()

    )

    pdr = (

        state["delivered"] / state["requests"] * 100

        if state["requests"] > 0

        else 0

    )

    throughput = (

        state["bytes_transmitted"] * 8 / total_seconds

        if total_seconds > 0

        else 0

    )

    collision_rate = (

        state["collisions"] / state["attempts"] * 100

        if state["attempts"] > 0

        else 0

    )

    average_latency = (

        state["total_latency"] / state["completed_packets"]

        if state["completed_packets"] > 0

        else 0

    )

    # This measures allocated channel-slots divided by
    # all available channel-slots. It is NOT a direct
    # measurement of spectrum occupancy.

    channel_utilization = (

        state["allocated_transmissions"]

        / (total_slots * 8) * 100

        if total_slots > 0

        else 0

    )

    fairness = calculate_jain_index(

        list(state["user_delivered"].values())

    )

    overall = {

        "algorithm": algorithm,

        "evaluation_slots": total_slots,

        "packet_requests": state["requests"],

        "delivered_packets": state["delivered"],

        "failed_packets": state["failed_packets"],

        "pending_packets": pending_packets,

        "packet_delivery_ratio_percent": round(
            pdr,
            4
        ),

        "transmission_attempts": state["attempts"],

        "successful_attempts": state["successful_attempts"],

        "failed_attempts": state["failed_attempts"],

        "collisions": state["collisions"],

        "collision_rate_percent": round(
            collision_rate,
            4
        ),

        "handovers": state["handovers"],

        "bytes_transmitted": state["bytes_transmitted"],

        "throughput_bps": round(
            throughput,
            4
        ),

        "average_latency_seconds": round(
            average_latency,
            4
        ),

        "channel_utilization_percent": round(
            channel_utilization,
            4
        ),

        "jain_delivery_fairness": round(
            fairness,
            5
        )

    }

    pd.DataFrame(
        [overall]
    ).to_csv(

        report_dir / "overall_metrics.csv",

        index=False

    )

    # --------------------------------------------------------
    # USER METRICS
    # --------------------------------------------------------

    user_records = []

    for user_id in SECONDARY_USER_IDS:

        delivered = state["user_delivered"][user_id]

        latency_values = state["user_latency"][user_id]

        user_records.append({

            "user_id": user_id,

            "packets_delivered": delivered,

            "pending_packets": len(
                state["queues"][user_id]
            ),

            "average_latency_seconds": (

                float(np.mean(latency_values))

                if latency_values

                else 0

            )

        })

    pd.DataFrame(
        user_records
    ).to_csv(

        report_dir / "user_metrics.csv",

        index=False

    )

    # --------------------------------------------------------
    # CHANNEL ALLOCATION METRICS
    # --------------------------------------------------------

    channel_records = []

    for channel_number in range(1, 9):

        channel_id = f"CH{channel_number:02d}"

        channel_records.append({

            "channel_id": channel_id,

            "allocation_count": state[
                "channel_allocations"
            ].get(channel_id, 0)

        })

    pd.DataFrame(
        channel_records
    ).to_csv(

        report_dir / "channel_allocation_metrics.csv",

        index=False

    )

    # --------------------------------------------------------
    # TIME-PERIOD METRICS
    # --------------------------------------------------------

    if not slots_df.empty:

        period_summary = slots_df.groupby(
            "time_period"
        ).agg(

            slots=("simulation_time", "count"),

            requests=("requests_generated", "sum"),

            delivered=("successful_packets", "sum"),

            attempts=("transmission_attempts", "sum"),

            collisions=("collisions", "sum"),

            bytes_transmitted=("bytes_transmitted", "sum")

        ).reset_index()

        period_summary.to_csv(

            report_dir / "time_period_metrics.csv",

            index=False

        )

    return overall


# ============================================================
# VALIDATE COMMON SCENARIO DATA
# ============================================================

def load_and_validate_scenarios():

    print("\nSTEP 1: LOADING COMMON SCENARIOS")

    if not SCENARIO_FILE.exists():

        raise FileNotFoundError(
            f"Scenario file not found: {SCENARIO_FILE}"
        )

    scenarios_df = pd.read_csv(
        SCENARIO_FILE
    )

    required_columns = [

        "simulation_time",

        "time_period",

        "generated_requests",

        "cooperative_decisions",

        "channel_conditions",

        "actual_channel_status"

    ]

    missing_columns = [

        column

        for column in required_columns

        if column not in scenarios_df.columns

    ]

    if missing_columns:

        raise ValueError(
            f"Missing scenario columns: {missing_columns}"
        )

    scenarios_df["simulation_time"] = pd.to_datetime(
        scenarios_df["simulation_time"]
    )

    if scenarios_df["simulation_time"].duplicated().any():

        raise ValueError(
            "Duplicate simulation timestamps found."
        )

    scenarios_df = scenarios_df.sort_values(
        "simulation_time"
    ).reset_index(drop=True)

    test_df = scenarios_df.loc[

        (scenarios_df["simulation_time"] >= TEST_START)

        &

        (scenarios_df["simulation_time"] < TEST_END)

    ].copy()

    if len(test_df) != EXPECTED_TEST_SLOTS:

        raise ValueError(

            f"Expected {EXPECTED_TEST_SLOTS} test slots, "

            f"found {len(test_df)}"

        )

    if test_df["simulation_time"].iloc[0] != TEST_START:

        raise ValueError(
            "Incorrect held-out test start timestamp."
        )

    expected_last_timestamp = (

        TEST_END

        - pd.Timedelta(seconds=SLOT_DURATION_SECONDS)

    )

    if (

        test_df["simulation_time"].iloc[-1]

        != expected_last_timestamp

    ):

        raise ValueError(
            "Incorrect held-out test end timestamp."
        )

    print("Total scenario rows:", len(scenarios_df))

    print("Held-out test rows:", len(test_df))

    print("Test start:", test_df["simulation_time"].iloc[0])

    print("Test end:", test_df["simulation_time"].iloc[-1])

    print("Scenario validation: PASSED")

    return test_df


# ============================================================
# INITIALIZE HMM FORECASTS
# ============================================================

def initialize_hmm_forecasts(
    hmm_predictor,
    channel_ids
):

    forecasts = {}

    for channel_id in channel_ids:

        forecasts[channel_id] = (

            hmm_predictor.predict_next_slot(
                channel_id
            )

        )

    return forecasts


# ============================================================
# MAIN COMPARISON SIMULATION
# ============================================================

def main():

    print("\n" + "=" * 75)

    print(" SPECTRIX - STANDARD BASELINE VS ASSM")

    print("=" * 75)

    # --------------------------------------------------------
    # STEP 1: LOAD HELD-OUT SCENARIOS
    # --------------------------------------------------------

    test_df = load_and_validate_scenarios()

    # --------------------------------------------------------
    # STEP 2: INITIALIZE ALLOCATORS
    # --------------------------------------------------------

    print("\nSTEP 2: INITIALIZING ALGORITHMS")

    baseline_allocator = BaselineAllocator(
        verbose=False
    )

    assm_allocator = ASSMAllocator(
        verbose=False
    )

    # HMM constructor trains on the first five days.
    # Its training-derived initial beliefs are used to
    # initialize the held-out prediction process.

    hmm_predictor = HMMPredictor()

    print("\nBoth algorithms initialized.")

    # --------------------------------------------------------
    # STEP 3: INITIALIZE SEPARATE QUEUES
    # --------------------------------------------------------

    print("\nSTEP 3: INITIALIZING ALGORITHM STATES")

    states = initialize_algorithm_state()

    # --------------------------------------------------------
    # STEP 4: INITIALIZE FIRST HMM FORECAST
    # --------------------------------------------------------

    print("\nSTEP 4: INITIALIZING HMM FORECASTS")

    first_decisions = parse_json(

        test_df.iloc[0]["cooperative_decisions"]

    )

    channel_ids = list(
        first_decisions.keys()
    )

    states["ASSM"]["hmm_forecasts"] = (

        initialize_hmm_forecasts(

            hmm_predictor,

            channel_ids

        )

    )

    print("Initial forecasts prepared from training-derived beliefs.")

    # --------------------------------------------------------
    # STEP 5: REPLAY THE SAME HELD-OUT SCENARIOS
    # --------------------------------------------------------

    print("\nSTEP 5: REPLAYING HELD-OUT SCENARIOS")

    test_slots = 0

    for index, (_, scenario) in enumerate(
        test_df.iterrows()
    ):

        simulation_time = pd.Timestamp(
            scenario["simulation_time"]
        )

        sensing_decisions = parse_json(
            scenario["cooperative_decisions"]
        )

        # ----------------------------------------------------
        # RUN BASELINE
        # ----------------------------------------------------

        process_algorithm_slot(

            "BASELINE",

            states["BASELINE"],

            scenario,

            baseline_allocator

        )

        # ----------------------------------------------------
        # RUN ASSM
        # ----------------------------------------------------

        process_algorithm_slot(

            "ASSM",

            states["ASSM"],

            scenario,

            assm_allocator

        )

        test_slots += 1

        # ----------------------------------------------------
        # UPDATE HMM AFTER CURRENT SLOT
        # ----------------------------------------------------

        # The current observation updates the belief state.
        # The resulting forecast is used in the NEXT slot.

        next_forecasts = (

            hmm_predictor.process_observations(
                sensing_decisions
            )

        )

        states["ASSM"]["hmm_forecasts"] = (
            next_forecasts
        )

        # ----------------------------------------------------
        # DAILY PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 8640 == 0:

            completed_test_days = (

                index + 1

            ) // 8640

            print(

                f"Test day {completed_test_days}/2 completed | "

                f"Timestamp: {simulation_time}"

            )

    # --------------------------------------------------------
    # STEP 6: VERIFY IDENTICAL REQUEST COUNTS
    # --------------------------------------------------------

    print("\nSTEP 6: VALIDATING COMMON TRAFFIC")

    baseline_requests = states[
        "BASELINE"
    ]["requests"]

    assm_requests = states[
        "ASSM"
    ]["requests"]

    print(
        "Baseline requests:",
        baseline_requests
    )

    print(
        "ASSM requests:",
        assm_requests
    )

    if baseline_requests != assm_requests:

        raise ValueError(
            "Request counts differ between algorithms."
        )

    print("Common request count: PASSED")

    # --------------------------------------------------------
    # STEP 7: SAVE RESULTS
    # --------------------------------------------------------

    print("\nSTEP 7: SAVING COMPARISON RESULTS")

    baseline_metrics = save_algorithm_results(

        "BASELINE",

        states["BASELINE"],

        test_slots

    )

    assm_metrics = save_algorithm_results(

        "ASSM",

        states["ASSM"],

        test_slots

    )

    # --------------------------------------------------------
    # STEP 8: SAVE COMPARISON TABLE
    # --------------------------------------------------------

    comparison_df = pd.DataFrame([

        baseline_metrics,

        assm_metrics

    ])

    comparison_df.to_csv(

        OUTPUT_DIR / "algorithm_comparison.csv",

        index=False

    )

    # --------------------------------------------------------
    # STEP 9: SAVE EXPERIMENT METADATA
    # --------------------------------------------------------

    metadata = {

        "project": "SPECTRIX",

        "comparison_type": (
            "Paired common-scenario replay"
        ),

        "baseline_algorithm": (
            "First available channel using cooperative sensing"
        ),

        "assm_algorithm": (
            "Cooperative sensing, HMM prediction and fuzzy allocation"
        ),

        "training_period": (
            "2026-11-02 to 2026-11-06"
        ),

        "held_out_test_start": str(TEST_START),

        "held_out_test_end_exclusive": str(TEST_END),

        "evaluation_slots": test_slots,

        "slot_duration_seconds": SLOT_DURATION_SECONDS,

        "packet_id_method": (
            "User ID + original creation timestamp + "
            "request position within the shared slot"
        ),

        "transmission_randomness": (
            "Deterministic SHA-256 keyed random draw "
            "by user, packet ID and attempt number"
        ),

        "traffic_requests_baseline": baseline_requests,

        "traffic_requests_assm": assm_requests,

        "actual_occupancy_use": (
            "Transmission outcome evaluation only"
        ),

        "pending_packets_included_in_pdr": True,

        "latency_population": (
            "Successfully completed packets only"
        )

    }

    with open(

        OUTPUT_DIR / "comparison_metadata.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # STEP 10: DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" FINAL ALGORITHM COMPARISON")

    print("=" * 75)

    display_columns = [

        "algorithm",

        "packet_requests",

        "delivered_packets",

        "failed_packets",

        "pending_packets",

        "packet_delivery_ratio_percent",

        "transmission_attempts",

        "collisions",

        "collision_rate_percent",

        "throughput_bps",

        "average_latency_seconds",

        "channel_utilization_percent",

        "jain_delivery_fairness"

    ]

    print(

        comparison_df[
            display_columns
        ].to_string(index=False)

    )

    # --------------------------------------------------------
    # STEP 11: FINAL VALIDATION
    # --------------------------------------------------------

    print("\nCOMPARISON VALIDATION")

    print(
        "Evaluation slots:",
        test_slots
    )

    print(
        "Expected slots:",
        EXPECTED_TEST_SLOTS
    )

    print(
        "Same request counts:",
        baseline_requests == assm_requests
    )

    print(
        "Separate algorithm queues:",
        True
    )

    print(
        "Common saved scenarios:",
        True
    )

    print(
        "Stable packet IDs:",
        True
    )

    print(
        "Deterministic transmission draws:",
        True
    )

    print(
        "Actual occupancy hidden from allocators:",
        True
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" BASELINE VS ASSM COMPARISON COMPLETED")

    print("=" * 75)

    print("\nResults directory:")

    print(OUTPUT_DIR)

    print("\nGenerated reports:")

    print("BASELINE/slot_results.csv")

    print("BASELINE/overall_metrics.csv")

    print("BASELINE/user_metrics.csv")

    print("BASELINE/channel_allocation_metrics.csv")

    print("BASELINE/time_period_metrics.csv")

    print("ASSM/slot_results.csv")

    print("ASSM/overall_metrics.csv")

    print("ASSM/user_metrics.csv")

    print("ASSM/channel_allocation_metrics.csv")

    print("ASSM/time_period_metrics.csv")

    print("algorithm_comparison.csv")

    print("comparison_metadata.json")


if __name__ == "__main__":

    main()
