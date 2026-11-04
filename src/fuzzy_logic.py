
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyLogicEngine:

    def __init__(self):

        print("Initializing Fuzzy Logic Engine...")

        # Input variables
        self.snr = ctrl.Antecedent(
            np.arange(-10, 31, 1),
            "snr"
        )

        self.occupancy = ctrl.Antecedent(
            np.arange(0, 1.01, 0.01),
            "occupancy"
        )

        self.interference = ctrl.Antecedent(
            np.arange(0, 101, 1),
            "interference"
        )

        # Output variable
        self.suitability = ctrl.Consequent(
            np.arange(0, 101, 1),
            "suitability"
        )

        self.define_membership_functions()
        self.define_rules()

        self.fuzzy_system = ctrl.ControlSystem(
            self.rules
        )

        print("Fuzzy Logic Engine initialized successfully!")

    def define_membership_functions(self):

        # SNR membership functions
        self.snr["low"] = fuzz.trimf(
            self.snr.universe, [-10, -10, 10]
        )

        self.snr["medium"] = fuzz.trimf(
            self.snr.universe, [0, 10, 20]
        )

        self.snr["high"] = fuzz.trimf(
            self.snr.universe, [10, 30, 30]
        )

        # Occupancy probability membership functions
        self.occupancy["low"] = fuzz.trimf(
            self.occupancy.universe, [0, 0, 0.4]
        )

        self.occupancy["medium"] = fuzz.trimf(
            self.occupancy.universe, [0.2, 0.5, 0.8]
        )

        self.occupancy["high"] = fuzz.trimf(
            self.occupancy.universe, [0.6, 1, 1]
        )

        # Interference membership functions
        self.interference["low"] = fuzz.trimf(
            self.interference.universe, [0, 0, 40]
        )

        self.interference["medium"] = fuzz.trimf(
            self.interference.universe, [20, 50, 80]
        )

        self.interference["high"] = fuzz.trimf(
            self.interference.universe, [60, 100, 100]
        )

        # Suitability output membership functions
        self.suitability["poor"] = fuzz.trimf(
            self.suitability.universe, [0, 0, 40]
        )

        self.suitability["moderate"] = fuzz.trimf(
            self.suitability.universe, [20, 50, 80]
        )

        self.suitability["good"] = fuzz.trimf(
            self.suitability.universe, [60, 100, 100]
        )

    def define_rules(self):

        rules = []

        # Rule 1: Low SNR produces poor suitability
        rules.append(
            ctrl.Rule(
                self.snr["low"],
                self.suitability["poor"]
            )
        )

        # Rule 2: High occupancy produces poor suitability
        rules.append(
            ctrl.Rule(
                self.occupancy["high"],
                self.suitability["poor"]
            )
        )

        # Rule 3: High interference produces poor suitability
        rules.append(
            ctrl.Rule(
                self.interference["high"],
                self.suitability["poor"]
            )
        )

        # Rule 4: Good SNR, low occupancy and low interference
        rules.append(
            ctrl.Rule(
                self.snr["high"]
                & self.occupancy["low"]
                & self.interference["low"],
                self.suitability["good"]
            )
        )

        # Rule 5: Medium conditions
        rules.append(
            ctrl.Rule(
                self.snr["medium"]
                & self.occupancy["medium"]
                & self.interference["medium"],
                self.suitability["moderate"]
            )
        )

        # Rule 6: High SNR and low interference
        rules.append(
            ctrl.Rule(
                self.snr["high"]
                & self.interference["low"],
                self.suitability["good"]
            )
        )

        # Rule 7: Low occupancy and medium interference
        rules.append(
            ctrl.Rule(
                self.occupancy["low"]
                & self.interference["medium"],
                self.suitability["moderate"]
            )
        )

        # Rule 8: Medium SNR and low occupancy
        rules.append(
            ctrl.Rule(
                self.snr["medium"]
                & self.occupancy["low"]
                & self.interference["low"],
                self.suitability["good"]
            )
        )

        # Rule 9: Low SNR or high interference
        rules.append(
            ctrl.Rule(
                self.snr["low"]
                | self.interference["high"],
                self.suitability["poor"]
            )
        )

        self.rules = rules

    def evaluate_channel(
        self,
        snr_value,
        occupancy_probability,
        interference_value
    ):

        simulation = ctrl.ControlSystemSimulation(
            self.fuzzy_system
        )

        simulation.input["snr"] = snr_value
        simulation.input["occupancy"] = occupancy_probability
        simulation.input["interference"] = interference_value

        simulation.compute()

        score = simulation.output["suitability"]

        return {
            "snr": snr_value,
            "occupancy_probability": occupancy_probability,
            "interference": interference_value,
            "suitability_score": round(float(score), 2)
        }


if __name__ == "__main__":

    fuzzy_engine = FuzzyLogicEngine()

    test_cases = [
        {
            "name": "Excellent Channel",
            "snr": 25,
            "occupancy": 0.05,
            "interference": 10
        },
        {
            "name": "Moderate Channel",
            "snr": 15,
            "occupancy": 0.50,
            "interference": 50
        },
        {
            "name": "Poor Channel",
            "snr": -5,
            "occupancy": 0.90,
            "interference": 90
        }
    ]

    print("\n" + "=" * 60)
    print(" SPECTRIX FUZZY CHANNEL EVALUATION")
    print("=" * 60)

    for test in test_cases:

        result = fuzzy_engine.evaluate_channel(
            test["snr"],
            test["occupancy"],
            test["interference"]
        )

        print(f"\nScenario: {test['name']}")
        print("SNR:", result["snr"], "dB")
        print(
            "Occupancy probability:",
            result["occupancy_probability"]
        )
        print("Interference:", result["interference"])
        print(
            "Suitability score:",
            result["suitability_score"],
            "/ 100"
        )
