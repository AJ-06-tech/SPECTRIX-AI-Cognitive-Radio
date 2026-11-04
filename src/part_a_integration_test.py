
from datetime import datetime, timedelta

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from hmm_predictor import HMMPredictor
from assm_allocator import ASSMAllocator
from channel_quality import ChannelQualityModel
from secondary_user import SecondaryUserEnvironment
from transmission_simulator import TransmissionSimulator


def get_time_period(simulation_time):

    hour = simulation_time.hour

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


def main():

    print("\n" + "=" * 70)
    print(" SPECTRIX PART A COMPLETE INTEGRATION TEST")
    print("=" * 70)

    # --------------------------------------------------
    # INITIALIZE ALL MODULES
    # --------------------------------------------------

    print("\nINITIALIZING MODULES...")

    channel_simulator = ChannelSimulator()
    sensing_engine = SpectrumSensing()
    hmm_predictor = HMMPredictor()
    allocator = ASSMAllocator()
    quality_model = ChannelQualityModel()

    secondary_environment = SecondaryUserEnvironment()
    transmission_simulator = TransmissionSimulator()

    # --------------------------------------------------
    # SELECT TEST TIMESTAMP
    # --------------------------------------------------

    test_time_string = "2026-11-07 09:00:00"

    test_time = datetime.strptime(
        test_time_string,
        "%Y-%m-%d %H:%M:%S"
    )

    completion_time = test_time + timedelta(seconds=10)

    time_period = get_time_period(test_time)

    print("\nSimulation timestamp:", test_time)
    print("Time period:", time_period)

    # --------------------------------------------------
    # STEP 1: GET ACTUAL CHANNEL STATUS
    # --------------------------------------------------

    print("\nSTEP 1: CHANNEL STATUS")

    actual_status = channel_simulator.get_channel_status(
        test_time_string
    )

    for channel_id, information in actual_status.items():

        print(
            channel_id,
            "->",
            information["status"]
        )

    # --------------------------------------------------
    # STEP 2: GENERATE CHANNEL QUALITY
    # --------------------------------------------------

    print("\nSTEP 2: CHANNEL QUALITY")

    channel_conditions = quality_model.generate_conditions(
        actual_status
    )

    quality_model.display_conditions(channel_conditions)

    # --------------------------------------------------
    # STEP 3: INDIVIDUAL SPECTRUM SENSING
    # --------------------------------------------------

    print("\nSTEP 3: INDIVIDUAL SPECTRUM SENSING")

    sensing_reports = sensing_engine.perform_sensing(
        actual_status
    )

    # --------------------------------------------------
    # STEP 4: COOPERATIVE SENSING
    # --------------------------------------------------

    print("\nSTEP 4: COOPERATIVE SPECTRUM SENSING")

    final_decisions = sensing_engine.cooperative_sensing(
        sensing_reports
    )

    sensing_decisions = {
        channel_id: result["decision"]
        for channel_id, result in final_decisions.items()
    }

    for channel_id, decision in sensing_decisions.items():

        print(channel_id, "->", decision)

    # --------------------------------------------------
    # STEP 5: HMM OCCUPANCY PREDICTION
    # --------------------------------------------------

    print("\nSTEP 5: HMM OCCUPANCY PREDICTION")

    hmm_results = hmm_predictor.process_observations(
        sensing_decisions
    )

    hmm_predictions = {
        channel_id: {
            "probability_busy": result["probability_busy"]
        }
        for channel_id, result in hmm_results.items()
    }

    # --------------------------------------------------
    # STEP 6: GENERATE SECONDARY USER REQUESTS
    # --------------------------------------------------

    print("\nSTEP 6: SECONDARY USER REQUESTS")

    generated_requests = secondary_environment.generate_requests(
        time_period,
        test_time_string
    )

    secondary_environment.display_status(
        test_time_string,
        generated_requests
    )

    requesting_users = (
        secondary_environment.get_requesting_users()
    )

    # Ensure the integration pipeline can be tested
    # even when random request generation produces none.
    if not requesting_users:

        print(
            "\nNo random requests generated."
            "\nCreating one controlled test request for SU01."
        )

        class TestRandomGenerator:

            def random(self):
                return 0.0

        test_user = secondary_environment.get_user("SU01")

        test_user.generate_request(
            time_period,
            TestRandomGenerator(),
            test_time_string
        )

        requesting_users = (
            secondary_environment.get_requesting_users()
        )

    print("\nUsers requesting channel allocation:")

    print(requesting_users)

    # --------------------------------------------------
    # STEP 7: ASSM CHANNEL ALLOCATION
    # --------------------------------------------------

    print("\nSTEP 7: ASSM CHANNEL ALLOCATION")

    allocations = allocator.allocate_channels(
        requesting_users,
        sensing_decisions,
        hmm_predictions,
        channel_conditions
    )

    allocator.display_allocations()

    # --------------------------------------------------
    # STEP 8: PACKET TRANSMISSION
    # --------------------------------------------------

    print("\nSTEP 8: PACKET TRANSMISSION")

    for user_id, allocation in allocations.items():

        if allocation["status"] != "ALLOCATED":

            print(
                user_id,
                "-> No channel available. Packet remains pending."
            )

            continue

        selected_channel = allocation["channel_id"]

        user = secondary_environment.get_user(user_id)

        # Allocate the selected channel to the user
        allocation_success = user.allocate_channel(
            selected_channel
        )

        if not allocation_success:

            print(
                user_id,
                "-> Could not allocate channel to user."
            )

            continue

        request = user.request_queue[0]

        # Obtain actual simulated channel occupancy.
        # External activity is also treated as busy for
        # this transmission safety check.
        actual_channel_busy = (
            actual_status[selected_channel]["status"]
            != "AVAILABLE"
        )

        channel_busy_state = int(actual_channel_busy)

        channel_snr = channel_conditions[
            selected_channel
        ]["snr"]

        print(
            f"\nTransmitting packet for {user_id}"
            f" on {selected_channel}"
        )

        transmission_result = (
            transmission_simulator.transmit_packet(
                user_id=user_id,
                channel_id=selected_channel,
                channel_busy_state=channel_busy_state,
                channel_snr=channel_snr,
                attempt_number=request["attempts"],
                simulation_time=test_time_string
            )
        )

        print("Transmission result:", transmission_result)

        # Update the Secondary User queue
        user.complete_transmission(
            transmission_result,
            completion_time.strftime("%Y-%m-%d %H:%M:%S")
        )

    # --------------------------------------------------
    # STEP 9: DISPLAY FINAL STATISTICS
    # --------------------------------------------------

    print("\nSTEP 9: FINAL STATISTICS")

    secondary_environment.display_statistics()

    transmission_simulator.display_statistics()

    # --------------------------------------------------
    # COMPLETION
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(" PART A INTEGRATION TEST COMPLETED")
    print("=" * 70)

    return {
        "allocations": allocations,
        "secondary_user_statistics": {
            user_id: user.get_statistics()
            for user_id, user in secondary_environment.users.items()
        },
        "transmission_statistics":
            transmission_simulator.get_statistics()
    }


if __name__ == "__main__":

    main()
