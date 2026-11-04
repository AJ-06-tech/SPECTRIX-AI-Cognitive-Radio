
import pandas as pd
from pathlib import Path

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from hmm_predictor import HMMPredictor
from assm_allocator import ASSMAllocator
from channel_quality import ChannelQualityModel
from secondary_user import SecondaryUserEnvironment
from transmission_simulator import TransmissionSimulator


# ============================================================
# SPECTRIX PART A - FULL SEVEN-DAY SIMULATION
# ============================================================

def get_time_period(timestamp):

    hour = timestamp.hour

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
# INITIALIZE HMM FORECASTS
# ============================================================

def initialize_hmm_forecasts(hmm_predictor, channel_ids):

    forecasts = {}

    for channel_id in channel_ids:

        forecasts[channel_id] = (
            hmm_predictor.predict_next_slot(channel_id)
        )

    return forecasts


# ============================================================
# SAVE SIMULATION RESULTS
# ============================================================

def save_results(
    slot_records,
    user_environment,
    transmission_simulator
):

    report_dir = Path("outputs") / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    slots_df = pd.DataFrame(slot_records)

    # --------------------------------------------------------
    # SAVE SLOT-WISE RESULTS
    # --------------------------------------------------------

    slots_df.to_csv(
        report_dir / "part_a_slot_results.csv",
        index=False
    )

    # --------------------------------------------------------
    # TRANSMISSION STATISTICS
    # --------------------------------------------------------

    transmission_stats = (
        transmission_simulator.get_statistics()
    )

    total_requests = sum(
        user.total_requests
        for user in user_environment.users.values()
    )

    total_pending = sum(
        user.pending_requests
        for user in user_environment.users.values()
    )

    total_delivered = (
        transmission_stats["total_packets_sent"]
    )

    total_bytes = (
        transmission_stats["total_bytes_transmitted"]
    )

    total_attempts = (
        transmission_stats["total_transmission_attempts"]
    )

    total_simulation_seconds = (
        len(slots_df) * 10
    )

    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------

    throughput_bps = (
        total_bytes * 8 / total_simulation_seconds
        if total_simulation_seconds > 0
        else 0
    )

    packet_delivery_ratio = (
        total_delivered / total_requests * 100
        if total_requests > 0
        else 0
    )

    collision_rate = (
        transmission_stats["total_collisions"]
        / total_attempts * 100
        if total_attempts > 0
        else 0
    )

    completed_requests = sum(
        user.completed_request_count
        for user in user_environment.users.values()
    )

    total_waiting_time = sum(
        user.total_waiting_time
        for user in user_environment.users.values()
    )

    average_latency = (
        total_waiting_time / completed_requests
        if completed_requests > 0
        else 0
    )

    allocated_transmissions = int(
        slots_df["allocated_users"].sum()
    )

    overall_metrics = {

        "simulation_slots": len(slots_df),

        "simulation_days":
            len(slots_df) * 10 / 86400,

        "total_packet_requests":
            total_requests,

        "successfully_delivered_packets":
            total_delivered,

        "pending_packets":
            total_pending,

        "packet_delivery_ratio_percent":
            round(packet_delivery_ratio, 3),

        "total_transmission_attempts":
            total_attempts,

        "successful_transmissions":
            transmission_stats["successful_transmissions"],

        "failed_packets":
            transmission_stats["failed_transmissions"],

        "failed_attempts":
            transmission_stats["failed_attempts"],

        "total_collisions":
            transmission_stats["total_collisions"],

        "collision_rate_percent":
            round(collision_rate, 3),

        "total_handovers":
            transmission_stats["total_handovers"],

        "total_bytes_transmitted":
            total_bytes,

        "throughput_bps":
            round(throughput_bps, 3),

        "average_packet_latency_seconds":
            round(average_latency, 3),

        "allocated_user_transmissions":
            allocated_transmissions
    }

    pd.DataFrame(
        [overall_metrics]
    ).to_csv(
        report_dir / "part_a_overall_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # TIME-PERIOD PERFORMANCE
    # --------------------------------------------------------

    period_summary = slots_df.groupby(
        "time_period"
    ).agg(

        slots=("simulation_time", "count"),

        requests_generated=("requests_generated", "sum"),

        users_allocated=("allocated_users", "sum"),

        successful_packets=("successful_packets", "sum"),

        collisions=("collisions", "sum"),

        attempts=("transmission_attempts", "sum"),

        bytes_transmitted=("bytes_transmitted", "sum")

    ).reset_index()

    period_summary.to_csv(
        report_dir / "part_a_time_period_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # INDIVIDUAL USER PERFORMANCE
    # --------------------------------------------------------

    user_metrics = []

    for user_id, user in user_environment.users.items():

        user_metrics.append(
            user.get_statistics()
        )

    pd.DataFrame(
        user_metrics
    ).to_csv(
        report_dir / "part_a_user_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # TRAINING AND HELD-OUT PERIOD SUMMARY
    # --------------------------------------------------------

    slots_df["evaluation_window"] = slots_df[
        "simulation_time"
    ].apply(

        lambda value:

        "Training/Warm-up Replay"

        if pd.Timestamp(value) < pd.Timestamp("2026-11-07")

        else "Held-out Test"
    )

    window_summary = slots_df.groupby(
        "evaluation_window"
    ).agg(

        slots=("simulation_time", "count"),

        requests_generated=("requests_generated", "sum"),

        allocated_users=("allocated_users", "sum"),

        successful_packets=("successful_packets", "sum"),

        collisions=("collisions", "sum"),

        transmission_attempts=("transmission_attempts", "sum"),

        bytes_transmitted=("bytes_transmitted", "sum")

    ).reset_index()

    window_summary.to_csv(
        report_dir / "part_a_evaluation_window_metrics.csv",
        index=False
    )

    print("\nAll reports saved successfully.")

    return overall_metrics


# ============================================================
# MAIN SIMULATION
# ============================================================

def main():

    print("\n" + "=" * 75)
    print(" SPECTRIX - PART A FULL SEVEN-DAY SIMULATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. INITIALIZE MODULES
    # --------------------------------------------------------

    print("\nSTEP 1: INITIALIZING MODULES")

    channel_simulator = ChannelSimulator()

    sensing_engine = SpectrumSensing()

    hmm_predictor = HMMPredictor()

    allocator = ASSMAllocator(
        verbose=False
    )

    quality_model = ChannelQualityModel()

    user_environment = SecondaryUserEnvironment()

    transmission_simulator = TransmissionSimulator()

    timestamps = channel_simulator.get_all_timestamps()

    total_slots = len(timestamps)

    if total_slots != 60480:

        raise ValueError(
            f"Expected 60480 slots, found {total_slots}"
        )

    print("\nAll modules initialized successfully.")

    print("Total simulation slots:", total_slots)

    # --------------------------------------------------------
    # 2. INITIALIZE HMM FORECASTS
    # --------------------------------------------------------

    print("\nSTEP 2: INITIALIZING HMM FORECASTS")

    initial_status = channel_simulator.get_channel_status(
        timestamps[0]
    )

    channel_ids = list(initial_status.keys())

    hmm_forecasts = initialize_hmm_forecasts(
        hmm_predictor,
        channel_ids
    )

    print("Initial channel forecasts prepared.")

    # --------------------------------------------------------
    # 3. INITIALIZE SIMULATION RECORDS
    # --------------------------------------------------------

    print("\nSTEP 3: STARTING SEVEN-DAY SIMULATION")

    slot_records = []

    total_requests_generated = 0
    total_allocated = 0
    total_successful = 0
    total_collisions = 0

    print("Simulation start:", timestamps[0])

    print("Simulation end:", timestamps[-1])

    # --------------------------------------------------------
    # 4. MAIN SIMULATION LOOP
    # --------------------------------------------------------

    for slot_index, simulation_time in enumerate(timestamps):

        time_period = get_time_period(
            pd.Timestamp(simulation_time)
        )

        # ----------------------------------------------------
        # A. READ CHANNEL STATUS
        # ----------------------------------------------------

        channel_status = channel_simulator.get_channel_status(
            simulation_time
        )

        # ----------------------------------------------------
        # B. GENERATE CHANNEL QUALITY
        # ----------------------------------------------------

        channel_conditions = quality_model.generate_conditions(
            channel_status
        )

        # ----------------------------------------------------
        # C. COOPERATIVE SPECTRUM SENSING
        # ----------------------------------------------------

        sensing_reports = sensing_engine.perform_sensing(
            channel_status
        )

        cooperative_results = (
            sensing_engine.cooperative_sensing(
                sensing_reports
            )
        )

        sensing_decisions = {

            channel_id: result["decision"]

            for channel_id, result in cooperative_results.items()
        }

        # ----------------------------------------------------
        # D. GENERATE SECONDARY USER REQUESTS
        # ----------------------------------------------------

        generated_requests = (
            user_environment.generate_requests(
                time_period,
                simulation_time
            )
        )

        total_requests_generated += len(
            generated_requests
        )

        # ----------------------------------------------------
        # E. INITIALIZE SLOT METRICS
        # ----------------------------------------------------

        requesting_users = (
            user_environment.get_requesting_users()
        )

        successful_packets_this_slot = 0

        collisions_this_slot = 0

        attempts_this_slot = 0

        bytes_this_slot = 0

        allocated_users_this_slot = 0

        # ----------------------------------------------------
        # F. ASSM CHANNEL ALLOCATION
        # ----------------------------------------------------

        if requesting_users:

            allocations = allocator.allocate_channels(

                requesting_users,

                sensing_decisions,

                hmm_forecasts,

                channel_conditions
            )

            # ------------------------------------------------
            # G. PACKET TRANSMISSION
            # ------------------------------------------------

            for user_id, allocation in allocations.items():

                if allocation["status"] != "ALLOCATED":
                    continue

                channel_id = allocation["channel_id"]

                user = user_environment.get_user(user_id)

                # Ensure that the user has a pending packet.
                if not user.request_queue:
                    continue

                # Retrieve the current packet before allocation.
                request = user.request_queue[0]

                previous_channel = request.get(
                    "last_channel"
                )

                # Allocate channel to the secondary user.
                if not user.allocate_channel(channel_id):
                    continue

                allocated_users_this_slot += 1

                total_allocated += 1

                # ------------------------------------------------
                # HANDOVER CORRECTION
                # ------------------------------------------------

                # A handover is counted only when the SAME
                # pending packet is retried on another channel.
                if (
                    request["attempts"] > 1

                    and previous_channel is not None

                    and previous_channel != channel_id
                ):

                    transmission_simulator.perform_handover(
                        user_id,
                        channel_id
                    )

                # Remember the channel used by this packet.
                request["last_channel"] = channel_id

                # ------------------------------------------------
                # ACTUAL TRANSMISSION
                # ------------------------------------------------

                # Ground truth is used only for the simulated
                # transmission outcome, never for ASSM selection.
                actual_status = channel_status[
                    channel_id
                ]["status"]

                channel_busy_state = (
                    0 if actual_status == "AVAILABLE" else 1
                )

                channel_snr = channel_conditions[
                    channel_id
                ]["snr"]

                attempt_number = request["attempts"]

                transmission_result = (
                    transmission_simulator.transmit_packet(

                        user_id=user_id,

                        channel_id=channel_id,

                        channel_busy_state=channel_busy_state,

                        channel_snr=channel_snr,

                        attempt_number=attempt_number,

                        simulation_time=simulation_time
                    )
                )

                attempts_this_slot += 1

                # ------------------------------------------------
                # UPDATE TRANSMISSION RESULTS
                # ------------------------------------------------

                if transmission_result["status"] == "SUCCESS":

                    successful_packets_this_slot += 1

                    bytes_this_slot += (
                        transmission_result["bytes_transmitted"]
                    )

                if transmission_result["status"] == "COLLISION":

                    collisions_this_slot += 1

                user.complete_transmission(

                    transmission_result,

                    pd.Timestamp(simulation_time)
                    + pd.Timedelta(seconds=10)
                )

        # ----------------------------------------------------
        # H. UPDATE HMM FOR THE NEXT SLOT
        # ----------------------------------------------------

        # Current sensing observations update the HMM.
        # The resulting prediction is used in the NEXT slot.

        next_forecasts = hmm_predictor.process_observations(
            sensing_decisions
        )

        hmm_forecasts = next_forecasts

        # ----------------------------------------------------
        # I. UPDATE GLOBAL METRICS
        # ----------------------------------------------------

        total_successful += successful_packets_this_slot

        total_collisions += collisions_this_slot

        # ----------------------------------------------------
        # J. RECORD SLOT RESULTS
        # ----------------------------------------------------

        slot_records.append({

            "simulation_time": simulation_time,

            "time_period": time_period,

            "requests_generated": len(generated_requests),

            "requesting_users": len(requesting_users),

            "allocated_users": allocated_users_this_slot,

            "transmission_attempts": attempts_this_slot,

            "successful_packets": successful_packets_this_slot,

            "collisions": collisions_this_slot,

            "bytes_transmitted": bytes_this_slot

        })

        # ----------------------------------------------------
        # K. DAILY PROGRESS
        # ----------------------------------------------------

        if (slot_index + 1) % 8640 == 0:

            completed_days = (
                slot_index + 1
            ) // 8640

            print(

                f"\nDay {completed_days}/7 completed | "

                f"Time: {simulation_time} | "

                f"Requests: {total_requests_generated} | "

                f"Allocated: {total_allocated} | "

                f"Delivered: {total_successful} | "

                f"Collisions: {total_collisions}"

            )

    # --------------------------------------------------------
    # 5. SAVE RESULTS
    # --------------------------------------------------------

    print("\nSTEP 4: SAVING SIMULATION RESULTS")

    overall_metrics = save_results(

        slot_records,

        user_environment,

        transmission_simulator
    )

    # --------------------------------------------------------
    # 6. DISPLAY FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" SPECTRIX PART A - FINAL SIMULATION RESULTS")

    print("=" * 75)

    for metric, value in overall_metrics.items():

        print(f"{metric}: {value}")

    # --------------------------------------------------------
    # 7. DISPLAY USER STATISTICS
    # --------------------------------------------------------

    print("\nFINAL SECONDARY USER STATISTICS")

    user_environment.display_statistics()

    # --------------------------------------------------------
    # 8. DISPLAY TRANSMISSION STATISTICS
    # --------------------------------------------------------

    print("\nFINAL TRANSMISSION STATISTICS")

    transmission_simulator.display_statistics()

    # --------------------------------------------------------
    # 9. COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" SEVEN-DAY SIMULATION COMPLETED SUCCESSFULLY")

    print("=" * 75)

    print("\nGenerated reports:")

    print("outputs/reports/part_a_slot_results.csv")

    print("outputs/reports/part_a_overall_metrics.csv")

    print("outputs/reports/part_a_time_period_metrics.csv")

    print("outputs/reports/part_a_user_metrics.csv")

    print("outputs/reports/part_a_evaluation_window_metrics.csv")


if __name__ == "__main__":

    main()
