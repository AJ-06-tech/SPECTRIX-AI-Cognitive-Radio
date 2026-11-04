
import pandas as pd

from hmm_predictor import HMMPredictor
from config import REPORT_DIR


def main():

    print("\nSPECTRIX HMM TRANSITION MATRIX ANALYSIS")
    print("=" * 55)

    # HMMPredictor automatically loads and trains its model
    predictor = HMMPredictor()

    results = []

    for channel_id, matrix in predictor.transition_matrices.items():

        print(f"\n{channel_id}")
        print("-" * 35)

        print("Transition Matrix:")
        print(matrix)

        results.append({
            "Channel": channel_id,
            "P(IDLE->IDLE)": matrix[0, 0],
            "P(IDLE->BUSY)": matrix[0, 1],
            "P(BUSY->IDLE)": matrix[1, 0],
            "P(BUSY->BUSY)": matrix[1, 1]
        })

    result_df = pd.DataFrame(results)

    output_path = REPORT_DIR / "hmm_transition_matrices.csv"
    result_df.to_csv(output_path, index=False)

    print("\nTRANSITION MATRIX SUMMARY")
    print(result_df.round(4).to_string(index=False))

    print("\nReport saved:")
    print(output_path)

    print("\nTRANSITION ANALYSIS COMPLETED!")


if __name__ == "__main__":
    main()
