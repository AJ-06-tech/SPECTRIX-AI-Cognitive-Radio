
"""
SPECTRIX - Robustness HMM Configuration and Validation
"""

import numpy as np

from hmm_predictor import HMMPredictor
from robustness_config import (
    get_experiment_config,
    SCALABILITY_USER_COUNTS
)


def create_robustness_hmm(user_count, verbose=False):

    settings = get_experiment_config(user_count)

    hmm = HMMPredictor()

    # Apply probabilities for the selected user count.
    hmm.detection_probability = float(
        settings["cooperative_detection_probability"]
    )

    hmm.false_alarm_probability = float(
        settings["cooperative_false_alarm_probability"]
    )

    if verbose:
        print("\nROBUSTNESS HMM CONFIGURATION")
        print("Secondary users:", user_count)
        print(
            "Detection probability:",
            hmm.detection_probability
        )
        print(
            "False alarm probability:",
            hmm.false_alarm_probability
        )

    return hmm


def validate_robustness_hmm(user_count):

    settings = get_experiment_config(user_count)

    hmm = create_robustness_hmm(
        user_count,
        verbose=False
    )

    expected_pd = settings[
        "cooperative_detection_probability"
    ]

    expected_pfa = settings[
        "cooperative_false_alarm_probability"
    ]

    # Verify configured probabilities.
    assert np.isclose(
        hmm.detection_probability,
        expected_pd
    )

    assert np.isclose(
        hmm.false_alarm_probability,
        expected_pfa
    )

    # Verify the actual observation likelihood calculations.
    idle_likelihood = hmm.observation_likelihood("IDLE")
    busy_likelihood = hmm.observation_likelihood("BUSY")

    expected_idle = np.array([
        1 - expected_pfa,
        1 - expected_pd
    ])

    expected_busy = np.array([
        expected_pfa,
        expected_pd
    ])

    assert np.allclose(
        idle_likelihood,
        expected_idle
    )

    assert np.allclose(
        busy_likelihood,
        expected_busy
    )

    print(
        f"{user_count} users | "
        f"Pd={expected_pd:.8f} | "
        f"Pfa={expected_pfa:.8f} | PASS"
    )


if __name__ == "__main__":

    print("\nSPECTRIX HMM ROBUSTNESS VALIDATION")
    print("=" * 65)

    for count in SCALABILITY_USER_COUNTS:
        validate_robustness_hmm(count)

    print("\nAll HMM robustness validations passed!")
