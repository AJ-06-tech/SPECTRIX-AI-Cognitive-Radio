
"""
SPECTRIX - Scalability Robustness Comparison
Paired comparison of Standard Baseline vs ASSM.

Run:
    python src/robustness_comparison.py --users 5
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    MAX_TRANSMISSION_ATTEMPTS,
    PACKET_SIZE_BYTES,
    CHANNEL_IDS
)

from hmm_predictor import HMMPredictor
from robustness_hmm import create_robustness_hmm
from assm_allocator import ASSMAllocator
from baseline_allocator import BaselineAllocator
from robustness_config import SCALABILITY_USER_COUNTS


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_START = pd.Timestamp("2026-11-07 00:00:00")
TEST_END = pd.Timestamp("2026-11-09 00:00:00")

EXPECTED_TEST_SLOTS = 17280
SLOT_SECONDS = 10
CHANNEL_COUNT = len(CHANNEL_IDS)

ALGORITHMS = ["BASELINE", "ASSM"]


def parse_json(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def deterministic_random(user_id, packet_id, attempt):
    key = f"{user_id}|{packet_id}|{attempt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    number = int.from_bytes(digest[:8], "big")
    return number / (2 ** 64)


def create_packet_id(user_id, created_at, position):
    timestamp = pd.Timestamp(created_at).isoformat()
    return f"{user_id}|{timestamp}|{position}"


def initialize_state(user_ids):
    states = {}

    for algorithm in ALGORITHMS:
        states[algorithm] = {
            "queues": {uid: [] for uid in user_ids},
            "requests": 0,
            "delivered": 0,
            "failed_packets": 0,
            "attempts": 0,
            "successful_attempts": 0,
            "failed_attempts": 0,
            "collisions": 0,
            "bytes_transmitted": 0,
            "total_latency": 0.0,
            "completed_packets": 0,
            "allocated_transmissions": 0,
            "user_delivered": {uid: 0 for uid in user_ids},
            "user_latency": {uid: [] for uid in user_ids},
            "channel_allocations": {},
            "slot_records": []
        }

    return states


def add_requests(state, generated_requests):
    count = 0

    for position, original in enumerate(generated_requests):
        request = copy.deepcopy(original)
        user_id = request["user_id"]

        if user_id not in state["queues"]:
            raise ValueError(f"Unknown user in scenario: {user_id}")

        request["attempts"] = 0
        request["last_channel"] = None
        request["packet_id"] = create_packet_id(
            user_id,
            request["created_at"],
            position
        )

        state["queues"][user_id].append(request)
        count += 1

    state["requests"] += count
    return count


def transmit_packet(user_id, request, channel_id, actual_status, snr):
    request["attempts"] += 1
    attempt = request["attempts"]

    if actual_status[channel_id]["status"] != "AVAILABLE":
        return {
            "status": "COLLISION",
            "failed": attempt >= MAX_TRANSMISSION_ATTEMPTS,
            "bytes": 0
        }

    probability = float(np.clip(0.50 + snr / 60.0, 0.10, 0.98))

    draw = deterministic_random(
        user_id,
        request["packet_id"],
        attempt
    )

    if draw < probability:
        return {
            "status": "SUCCESS",
            "failed": False,
            "bytes": request.get("packet_size", PACKET_SIZE_BYTES)
        }

    return {
        "status": "FAILED",
        "failed": attempt >= MAX_TRANSMISSION_ATTEMPTS,
        "bytes": 0
    }


def process_slot(algorithm, state, scenario, allocator, user_ids):
    timestamp = pd.Timestamp(scenario["simulation_time"])

    actual_status = parse_json(scenario["actual_channel_status"])
    conditions = parse_json(scenario["channel_conditions"])
    decisions = parse_json(scenario["cooperative_decisions"])
    generated = parse_json(scenario["generated_requests"])

    new_requests = add_requests(state, generated)

    requesting_users = [
        uid for uid in user_ids if state["queues"][uid]
    ]

    if algorithm == "BASELINE":
        allocations = allocator.allocate_channels(
            requesting_users,
            decisions
        )
    elif requesting_users:
        allocations = allocator.allocate_channels(
            requesting_users,
            decisions,
            state["hmm_forecasts"],
            conditions
        )
    else:
        allocations = {}

    delivered_slot = 0
    collisions_slot = 0
    attempts_slot = 0
    bytes_slot = 0
    allocated_slot = 0

    for user_id, allocation in allocations.items():
        if allocation["status"] != "ALLOCATED":
            continue

        channel_id = allocation["channel_id"]
        queue = state["queues"][user_id]

        if not queue:
            continue

        request = queue[0]

        previous_channel = request["last_channel"]
        if previous_channel is not None and previous_channel != channel_id:
            if request["attempts"] > 0:
                state["handovers"] = state.get("handovers", 0) + 1

        request["last_channel"] = channel_id

        snr = conditions[channel_id]["snr"]

        result = transmit_packet(
            user_id,
            request,
            channel_id,
            actual_status,
            snr
        )

        attempts_slot += 1
        allocated_slot += 1
        state["attempts"] += 1
        state["allocated_transmissions"] += 1

        state["channel_allocations"][channel_id] = (
            state["channel_allocations"].get(channel_id, 0) + 1
        )

        if result["status"] == "SUCCESS":
            delivered_slot += 1
            state["delivered"] += 1
            state["successful_attempts"] += 1
            state["bytes_transmitted"] += result["bytes"]
            bytes_slot += result["bytes"]

            completion = timestamp + pd.Timedelta(seconds=SLOT_SECONDS)
            created = pd.Timestamp(request["created_at"])
            latency = (completion - created).total_seconds()

            state["total_latency"] += latency
            state["completed_packets"] += 1
            state["user_delivered"][user_id] += 1
            state["user_latency"][user_id].append(latency)

            queue.pop(0)

        elif result["status"] == "COLLISION":
            collisions_slot += 1
            state["collisions"] += 1
            state["failed_attempts"] += 1

            if result["failed"]:
                state["failed_packets"] += 1
                queue.pop(0)

        else:
            state["failed_attempts"] += 1

            if result["failed"]:
                state["failed_packets"] += 1
                queue.pop(0)

    state["slot_records"].append({
        "simulation_time": str(timestamp),
        "time_period": scenario["time_period"],
        "requests_generated": new_requests,
        "requesting_users": len(requesting_users),
        "allocated_users": allocated_slot,
        "transmission_attempts": attempts_slot,
        "successful_packets": delivered_slot,
        "collisions": collisions_slot,
        "bytes_transmitted": bytes_slot
    })


def jain_index(values):
    values = np.asarray(values, dtype=float)

    denominator = len(values) * np.sum(values ** 2)

    if denominator == 0:
        return 0.0

    return float(np.sum(values) ** 2 / denominator)


def save_results(algorithm, state, user_ids, output_dir):
    report_dir = output_dir / algorithm
    report_dir.mkdir(parents=True, exist_ok=True)

    slots = pd.DataFrame(state["slot_records"])
    slots.to_csv(report_dir / "slot_results.csv", index=False)

    total_slots = len(slots)
    total_seconds = total_slots * SLOT_SECONDS

    pending = sum(len(q) for q in state["queues"].values())

    pdr = (
        state["delivered"] / state["requests"] * 100
        if state["requests"] else 0
    )

    throughput = (
        state["bytes_transmitted"] * 8 / total_seconds
        if total_seconds else 0
    )

    collision_rate = (
        state["collisions"] / state["attempts"] * 100
        if state["attempts"] else 0
    )

    latency = (
        state["total_latency"] / state["completed_packets"]
        if state["completed_packets"] else 0
    )

    utilization = (
        state["allocated_transmissions"]
        / (total_slots * CHANNEL_COUNT) * 100
        if total_slots else 0
    )

    overall = {
        "algorithm": algorithm,
        "secondary_users": len(user_ids),
        "evaluation_slots": total_slots,
        "packet_requests": state["requests"],
        "delivered_packets": state["delivered"],
        "failed_packets": state["failed_packets"],
        "pending_packets": pending,
        "packet_delivery_ratio_percent": round(pdr, 4),
        "transmission_attempts": state["attempts"],
        "successful_attempts": state["successful_attempts"],
        "failed_attempts": state["failed_attempts"],
        "collisions": state["collisions"],
        "collision_rate_percent": round(collision_rate, 4),
        "throughput_bps": round(throughput, 4),
        "average_latency_seconds": round(latency, 4),
        "channel_utilization_percent": round(utilization, 4),
        "jain_delivery_fairness": round(
            jain_index(list(state["user_delivered"].values())),
            5
        )
    }

    pd.DataFrame([overall]).to_csv(
        report_dir / "overall_metrics.csv",
        index=False
    )

    user_rows = []

    for uid in user_ids:
        values = state["user_latency"][uid]

        user_rows.append({
            "user_id": uid,
            "packets_delivered": state["user_delivered"][uid],
            "pending_packets": len(state["queues"][uid]),
            "average_latency_seconds": (
                float(np.mean(values)) if values else 0
            )
        })

    pd.DataFrame(user_rows).to_csv(
        report_dir / "user_metrics.csv",
        index=False
    )

    channel_rows = [
        {
            "channel_id": channel,
            "allocation_count": state["channel_allocations"].get(channel, 0)
        }
        for channel in CHANNEL_IDS
    ]

    pd.DataFrame(channel_rows).to_csv(
        report_dir / "channel_allocation_metrics.csv",
        index=False
    )

    if not slots.empty:
        period = slots.groupby("time_period").agg(
            slots=("simulation_time", "count"),
            requests=("requests_generated", "sum"),
            delivered=("successful_packets", "sum"),
            attempts=("transmission_attempts", "sum"),
            collisions=("collisions", "sum"),
            bytes_transmitted=("bytes_transmitted", "sum")
        ).reset_index()

        period.to_csv(
            report_dir / "time_period_metrics.csv",
            index=False
        )

    return overall


def discover_user_ids(scenarios_df):
    found = set()

    for value in scenarios_df["generated_requests"]:
        for request in parse_json(value):
            found.add(request["user_id"])

    if not found:
        raise ValueError("No secondary user IDs found in generated requests.")

    return sorted(found)


def run_experiment(user_count):
    print("\n" + "=" * 75)
    print(f"SPECTRIX SCALABILITY COMPARISON - {user_count} USERS")
    print("=" * 75)

    scenario_dir = (
        BASE_DIR / "data" / "processed"
        / "robustness_testing" / f"users_{user_count}"
    )

    scenario_file = scenario_dir / "common_scenarios.csv"
    metadata_file = scenario_dir / "scenario_metadata.json"

    if not scenario_file.exists():
        raise FileNotFoundError(scenario_file)

    if not metadata_file.exists():
        raise FileNotFoundError(metadata_file)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    scenarios = pd.read_csv(scenario_file)
    scenarios["simulation_time"] = pd.to_datetime(
        scenarios["simulation_time"]
    )

    if len(scenarios) != 60480:
        raise ValueError(
            f"Expected 60480 scenario rows; found {len(scenarios)}"
        )

    user_ids = discover_user_ids(scenarios)

    if len(user_ids) != user_count:
        raise ValueError(
            f"Expected {user_count} users, found {len(user_ids)}: {user_ids}"
        )

    test = scenarios.loc[
        (scenarios["simulation_time"] >= TEST_START)
        & (scenarios["simulation_time"] < TEST_END)
    ].copy().reset_index(drop=True)

    if len(test) != EXPECTED_TEST_SLOTS:
        raise ValueError(
            f"Expected {EXPECTED_TEST_SLOTS} test slots; found {len(test)}"
        )

    print("Scenario rows:", len(scenarios))
    print("Secondary users:", user_ids)
    print("Held-out slots:", len(test))

    output_dir = (
        BASE_DIR / "outputs" / "reports"
        / "robustness_testing" / f"users_{user_count}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = BaselineAllocator(verbose=False)
    assm = ASSMAllocator(verbose=False)

    hmm = create_robustness_hmm(user_count, verbose=True)

    first_decisions = parse_json(
        test.iloc[0]["cooperative_decisions"]
    )

    states = initialize_state(user_ids)

    states["ASSM"]["hmm_forecasts"] = {
        channel: hmm.predict_next_slot(channel)
        for channel in first_decisions
    }

    print("\nReplaying held-out scenarios...")

    for index, (_, scenario) in enumerate(test.iterrows()):
        decisions = parse_json(scenario["cooperative_decisions"])

        process_slot(
            "BASELINE",
            states["BASELINE"],
            scenario,
            baseline,
            user_ids
        )

        process_slot(
            "ASSM",
            states["ASSM"],
            scenario,
            assm,
            user_ids
        )

        # Update HMM after the current slot for the next slot.
        states["ASSM"]["hmm_forecasts"] = (
            hmm.process_observations(decisions)
        )

        if (index + 1) % 8640 == 0:
            print(f"Evaluation day {(index + 1) // 8640}/2 completed.")

    baseline_requests = states["BASELINE"]["requests"]
    assm_requests = states["ASSM"]["requests"]

    if baseline_requests != assm_requests:
        raise ValueError("Baseline and ASSM request counts differ.")

    baseline_metrics = save_results(
        "BASELINE", states["BASELINE"], user_ids, output_dir
    )

    assm_metrics = save_results(
        "ASSM", states["ASSM"], user_ids, output_dir
    )

    comparison = pd.DataFrame([
        baseline_metrics,
        assm_metrics
    ])

    comparison.to_csv(
        output_dir / "algorithm_comparison.csv",
        index=False
    )

    result_metadata = {
        "project": "SPECTRIX",
        "experiment": "Scalability robustness",
        "secondary_users": user_count,
        "user_ids": user_ids,
        "scenario_file": str(scenario_file),
        "comparison_type": "Paired common-scenario replay",
        "training_period": "2026-11-02 to 2026-11-06",
        "held_out_period": "2026-11-07 to 2026-11-08",
        "evaluation_slots": len(test),
        "cooperative_detection_probability": hmm.detection_probability,
        "cooperative_false_alarm_probability": hmm.false_alarm_probability,
        "baseline_requests": baseline_requests,
        "assm_requests": assm_requests,
        "same_saved_scenarios": True,
        "separate_packet_queues": True,
        "deterministic_transmission_draws": True,
        "ground_truth_used_only_for_transmission_outcome": True,
        "source_scenario_metadata": metadata
    }

    with open(
        output_dir / "comparison_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(result_metadata, f, indent=4, default=str)

    print("\nFINAL COMPARISON")
    print(comparison.to_string(index=False))

    print("\nValidation:")
    print("Same request count:", baseline_requests == assm_requests)
    print("Correct user count:", len(user_ids) == user_count)
    print("Held-out slots:", len(test) == EXPECTED_TEST_SLOTS)
    print("Output:", output_dir)

    print("\nSCALABILITY EXPERIMENT COMPLETED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--users",
        type=int,
        required=True,
        choices=SCALABILITY_USER_COUNTS
    )

    args = parser.parse_args()
    run_experiment(args.users)
