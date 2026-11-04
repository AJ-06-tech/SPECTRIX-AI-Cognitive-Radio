from channel_simulator import ChannelSimulator
from robustness_spectrum_sensing import RobustnessSpectrumSensing

channel_simulator = ChannelSimulator()

test_time = "2026-11-07 09:00:00"

actual_status = channel_simulator.get_channel_status(test_time)

print("\nSPECTRUM SENSING FUNCTIONAL TEST")

for count in [3, 5, 8, 10]:

    engine = RobustnessSpectrumSensing(
        user_count=count,
        verbose=False
    )

    reports = engine.perform_sensing(actual_status)

    decisions = engine.cooperative_sensing(reports)

    assert len(reports) == count

    assert len(decisions) == 8

    for channel_id, result in decisions.items():

        assert result["busy_votes"] + result["idle_votes"] == count

        assert result["decision"] in ["BUSY", "IDLE"]

    print(
        f"{count} users | "
        f"Reports: {len(reports)} | "
        f"Channels: {len(decisions)} | "
        f"Voting threshold: {engine.required_votes} | "
        "PASS"
    )

print("\nAll sensing functional tests passed!")