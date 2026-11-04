
import pandas as pd

from channel_simulator import ChannelSimulator
from spectrum_sensing import SpectrumSensing
from hmm_predictor import HMMPredictor
from assm_allocator import ASSMAllocator
from channel_quality import ChannelQualityModel
from secondary_user import SecondaryUser, SecondaryUserEnvironment
from transmission_simulator import TransmissionSimulator


def generate_test_request(user, environment, simulation_time):

    # Generate one request reliably for testing.
    # This avoids the test depending on a single random request draw.
    for _ in range(100):

        request = user.generate_request(
            "Evening",
            environment.rng,
            simulation_time
        )

        if request is not None:
            return request

    raise RuntimeError(
        f"Could not generate a request for {user.user_id}"
    )


def main():

    print("\n" + "=" * 70)
    print(" SPECTRIX - PART A TRANSMISSION INTEGRATION TEST")
    print("=" * 70)

    # -------------------------------------------------
    # 1. INITIALIZE MODULES
    # -------------------------------------------------

    print("\nSTEP 1: INITIALIZING MODULES")

    channel_simulator = ChannelSimulator()
    sensing_engine = SpectrumSensing()
    hmm_predictor = HMMPredictor()
    allocator = ASSMAllocator(verbose=False)
    quality_model = ChannelQualityModel()
    user_environment = SecondaryUserEnvironment()
    transmission_simulator = TransmissionSimulator()

    # -------------------------------------------------
    # 2. SELECT TEST TIMESTAMP
    # -------------------------------------------------

    print("\nSTEP 2: SELECTING TEST TIMESTAMP")

    timestamps = channel_simulator.get_all_timestamps()

    test_time = timestamps[5 * 8640]

    completion_time = pd.Timestamp(test_time) + pd.Timedelta(
        seconds=10
    )

    print("Simulation time:", test_time)

    # -------------------------------------------------
    # 3. READ CHANNEL CONDITIONS
    # -------------------------------------------------

    print("\nSTEP 3: READING CHANNEL CONDITIONS")

    channel_status = channel_simulator.get_channel_status(
        test_time
    )

    channel_conditions = quality_model.generate_conditions(
        channel_status
    )

    # -------------------------------------------------
    # 4. COOPERATIVE SENSING
    # -------------------------------------------------

    print("\nSTEP 4: COOPERATIVE SENSING")

    sensing_reports = sensing_engine.perform_sensing(
        channel_status
    )

    cooperative_results = sensing_engine.cooperative_sensing(
        sensing_reports
    )

    sensing_decisions = {
        channel_id: result["decision"]
        for channel_id, result in cooperative_results.items()
    }

    # -------------------------------------------------
    # 5. HMM PREDICTION
    # -------------------------------------------------

    print("\nSTEP 5: HMM PREDICTION")

    hmm_predictions = hmm_predictor.process_observations(
        sensing_decisions
    )

    # -------------------------------------------------
    # 6. GENERATE PACKET REQUESTS
    # -------------------------------------------------

    print("\nSTEP 6: GENERATING SECONDARY USER REQUESTS")

    for user in user_environment.users.values():

        request = generate_test_request(
            user,
            user_environment,
            test_time
        )

        print(
            user.user_id,
            "generated a packet:",
            request["packet_size"],
            "bytes"
        )

    requesting_users = user_environment.get_requesting_users()

    print("\nUsers requesting channels:", requesting_users)

    # -------------------------------------------------
    # 7. ASSM CHANNEL ALLOCATION
    # -------------------------------------------------

    print("\nSTEP 7: ASSM CHANNEL ALLOCATION")

    allocations = allocator.allocate_channels(
        requesting_users,
        sensing_decisions,
        hmm_predictions,
        channel_conditions
    )

    for user_id, allocation in allocations.items():

        print(
            user_id,
            "->",
            allocation["status"],
            "| Channel:",
            allocation["channel_id"]
        )

    # -------------------------------------------------
    # 8. TRANSMIT ALLOCATED PACKETS
    # -------------------------------------------------

    print("\nSTEP 8: PACKET TRANSMISSION")

    for user_id, allocation in allocations.items():

        if allocation["status"] != "ALLOCATED":
            continue

        channel_id = allocation["channel_id"]

        user = user_environment.get_user(user_id)

        allocated = user.allocate_channel(channel_id)

        if not allocated:
            continue

        # Ground truth is used only to simulate the actual
        # transmission outcome, not for channel allocation.
        actual_status = channel_status[channel_id]["status"]

        channel_busy_state = (
            1 if actual_status != "AVAILABLE" else 0
        )

        snr = channel_conditions[channel_id]["snr"]

        request = user.request_queue[0]

        attempt_number = request["attempts"]

        result = transmission_simulator.transmit_packet(
            user_id=user_id,
            channel_id=channel_id,
            channel_busy_state=channel_busy_state,
            channel_snr=snr,
            attempt_number=attempt_number,
            simulation_time=test_time
        )

        user.complete_transmission(
            result,
            completion_time
        )

        print(
            user_id,
            "| Channel:",
            channel_id,
            "| Result:",
            result["status"],
            "| Attempt:",
            attempt_number
        )

    # -------------------------------------------------
    # 9. TEST COLLISION AND RETRY
    # -------------------------------------------------

    print("\nSTEP 9: TESTING COLLISION AND RETRY")

    retry_user = SecondaryUser("TEST_SU")

    # Create one packet for the retry test.
    retry_request = None

    for _ in range(100):

        retry_request = retry_user.generate_request(
            "Evening",
            user_environment.rng,
            test_time
        )

        if retry_request is not None:
            break

    if retry_request is None:
        raise RuntimeError("Could not create retry test packet")

    # First attempt: CH01 is occupied at the test timestamp.
    retry_user.allocate_channel("CH01")

    first_attempt = transmission_simulator.transmit_packet(
        user_id="TEST_SU",
        channel_id="CH01",
        channel_busy_state=1,
        channel_snr=18,
        attempt_number=1,
        simulation_time=test_time
    )

    retry_user.complete_transmission(
        first_attempt,
        completion_time
    )

    print(
        "First attempt:",
        first_attempt["status"],
        "| Retry required:",
        first_attempt["retry_required"]
    )

    # Second attempt: hand over to available CH06.
    handover_status = {
        "CH06": channel_status["CH06"]
    }

    handover_success = transmission_simulator.perform_handover(
        user_id="TEST_SU",
        new_channel="CH06",
        channel_status=handover_status
    )

    print("Handover to CH06:", handover_success)

    if handover_success:

        retry_user.allocate_channel("CH06")

        second_attempt = transmission_simulator.transmit_packet(
            user_id="TEST_SU",
            channel_id="CH06",
            channel_busy_state=0,
            channel_snr=30,
            attempt_number=2,
            simulation_time=completion_time
        )

        retry_user.complete_transmission(
            second_attempt,
            pd.Timestamp(completion_time) + pd.Timedelta(
                seconds=10
            )
        )

        print(
            "Second attempt:",
            second_attempt["status"],
            "| Attempt number:",
            second_attempt["attempt_number"]
        )

    # -------------------------------------------------
    # 10. DISPLAY FINAL STATISTICS
    # -------------------------------------------------

    print("\nSTEP 10: FINAL TRANSMISSION STATISTICS")

    transmission_simulator.display_statistics()

    print("\nSECONDARY USER STATISTICS")

    user_environment.display_statistics()

    print("\nRETRY TEST USER STATISTICS")

    print(retry_user.get_statistics())

    # -------------------------------------------------
    # COMPLETION
    # -------------------------------------------------

    print("\n" + "=" * 70)
    print(" TRANSMISSION INTEGRATION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
