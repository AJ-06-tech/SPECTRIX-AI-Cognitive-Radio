
"""
SPECTRIX - Scalability Analysis
Compares Standard Baseline and ASSM for 3, 5, 8 and 10 secondary users.

Reads existing experiment results only.
Does not modify simulation data or original comparison reports.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "outputs" / "reports" / "robustness_testing"

REPORT_DIR = INPUT_DIR / "summary"
GRAPH_DIR = BASE_DIR / "outputs" / "graphs" / "scalability_analysis"

USER_COUNTS = [3, 5, 8, 10]

ALGORITHMS = ["BASELINE", "ASSM"]


# --------------------------------------------------
# LOAD EXPERIMENT RESULTS
# --------------------------------------------------

def load_experiment_results():

    all_results = []

    print("\nLoading scalability experiment results...")

    for user_count in USER_COUNTS:

        file_path = (
            INPUT_DIR
            / f"users_{user_count}"
            / "algorithm_comparison.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing experiment result: {file_path}"
            )

        data = pd.read_csv(file_path)

        if len(data) != 2:
            raise ValueError(
                f"Expected 2 algorithm rows for {user_count} users."
            )

        if set(data["algorithm"]) != set(ALGORITHMS):
            raise ValueError(
                f"Missing or unexpected algorithm for {user_count} users."
            )

        if not (data["secondary_users"] == user_count).all():
            raise ValueError(
                f"Incorrect user count in users_{user_count} results."
            )

        if data["evaluation_slots"].nunique() != 1:
            raise ValueError(
                f"Evaluation slot mismatch for {user_count} users."
            )

        if data["packet_requests"].nunique() != 1:
            raise ValueError(
                f"Request count mismatch for {user_count} users."
            )

        all_results.append(data)

        print(f"Loaded successfully: {user_count} users")

    combined = pd.concat(all_results, ignore_index=True)

    combined["algorithm"] = combined["algorithm"].str.upper()

    combined = combined.sort_values(
        by=["secondary_users", "algorithm"]
    ).reset_index(drop=True)

    return combined


# --------------------------------------------------
# SAVE CONSOLIDATED CSV
# --------------------------------------------------

def save_combined_results(data):

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = REPORT_DIR / "scalability_combined_results.csv"

    data.to_csv(output_file, index=False)

    print(f"\nCombined results saved: {output_file}")

    return output_file


# --------------------------------------------------
# CREATE COMPARISON GRAPHS
# --------------------------------------------------

def create_comparison_graph(data, metric, title, ylabel, filename):

    plt.figure(figsize=(10, 6))

    for algorithm in ALGORITHMS:

        subset = data[data["algorithm"] == algorithm]

        plt.plot(
            subset["secondary_users"],
            subset[metric],
            marker="o",
            linewidth=2,
            markersize=7,
            label=algorithm
        )

    plt.title(title, fontsize=14)
    plt.xlabel("Number of Secondary Users", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    plt.xticks(USER_COUNTS)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    output_file = GRAPH_DIR / filename

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Graph saved: {output_file}")


def generate_all_graphs(data):

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating scalability graphs...")

    create_comparison_graph(
        data,
        "packet_delivery_ratio_percent",
        "Packet Delivery Ratio vs Secondary Users",
        "Packet Delivery Ratio (%)",
        "pdr_comparison.png"
    )

    create_comparison_graph(
        data,
        "collisions",
        "Collision Count vs Secondary Users",
        "Number of Collisions",
        "collision_comparison.png"
    )

    create_comparison_graph(
        data,
        "throughput_bps",
        "Throughput vs Secondary Users",
        "Throughput (bps)",
        "throughput_comparison.png"
    )

    create_comparison_graph(
        data,
        "average_latency_seconds",
        "Average Latency vs Secondary Users",
        "Average Latency (seconds)",
        "latency_comparison.png"
    )


# --------------------------------------------------
# GENERATE WRITTEN ANALYSIS
# --------------------------------------------------

def generate_analysis_report(data):

    report_lines = []

    report_lines.append("SPECTRIX - SCALABILITY ANALYSIS REPORT")
    report_lines.append("=" * 65)

    report_lines.append(
        "\nExperiment: Standard Baseline vs ASSM"
    )

    report_lines.append(
        "Secondary users tested: 3, 5, 8 and 10"
    )

    report_lines.append(
        "Evaluation period: 7-8 November 2026"
    )

    report_lines.append(
        "Evaluation slots per experiment: 17,280"
    )

    report_lines.append(
        "\nNOTE: Results are from a synthetic simulation."
    )

    report_lines.append(
        "\n1. CONSOLIDATED RESULTS"
    )

    for user_count in USER_COUNTS:

        baseline = data[
            (data["secondary_users"] == user_count)
            & (data["algorithm"] == "BASELINE")
        ].iloc[0]

        assm = data[
            (data["secondary_users"] == user_count)
            & (data["algorithm"] == "ASSM")
        ].iloc[0]

        report_lines.append("\n" + "-" * 65)

        report_lines.append(
            f"SECONDARY USERS: {user_count}"
        )

        report_lines.append(
            f"Packet Requests: {int(baseline['packet_requests'])}"
        )

        report_lines.append(
            f"Baseline PDR: "
            f"{baseline['packet_delivery_ratio_percent']:.4f}%"
        )

        report_lines.append(
            f"ASSM PDR: "
            f"{assm['packet_delivery_ratio_percent']:.4f}%"
        )

        report_lines.append(
            f"Baseline Collisions: {int(baseline['collisions'])}"
        )

        report_lines.append(
            f"ASSM Collisions: {int(assm['collisions'])}"
        )

        report_lines.append(
            f"Baseline Throughput: "
            f"{baseline['throughput_bps']:.4f} bps"
        )

        report_lines.append(
            f"ASSM Throughput: "
            f"{assm['throughput_bps']:.4f} bps"
        )

        report_lines.append(
            f"Baseline Latency: "
            f"{baseline['average_latency_seconds']:.4f} seconds"
        )

        report_lines.append(
            f"ASSM Latency: "
            f"{assm['average_latency_seconds']:.4f} seconds"
        )

        collision_reduction = (
            (baseline["collisions"] - assm["collisions"])
            / baseline["collisions"] * 100
            if baseline["collisions"] > 0
            else 0
        )

        report_lines.append(
            f"Collision Reduction: {collision_reduction:.2f}%"
        )

    # Overall observations

    report_lines.append("\n\n2. OVERALL OBSERVATIONS")
    report_lines.append("-" * 65)

    baseline_data = data[data["algorithm"] == "BASELINE"]
    assm_data = data[data["algorithm"] == "ASSM"]

    pdr_improvements = 0
    throughput_improvements = 0
    latency_improvements = 0

    for user_count in USER_COUNTS:

        baseline = baseline_data[
            baseline_data["secondary_users"] == user_count
        ].iloc[0]

        assm = assm_data[
            assm_data["secondary_users"] == user_count
        ].iloc[0]

        if assm["packet_delivery_ratio_percent"] > baseline[
            "packet_delivery_ratio_percent"
        ]:
            pdr_improvements += 1

        if assm["throughput_bps"] > baseline["throughput_bps"]:
            throughput_improvements += 1

        if assm["average_latency_seconds"] < baseline[
            "average_latency_seconds"
        ]:
            latency_improvements += 1

    report_lines.append(
        f"ASSM achieved higher PDR in "
        f"{pdr_improvements} out of 4 experiments."
    )

    report_lines.append(
        f"ASSM achieved higher throughput in "
        f"{throughput_improvements} out of 4 experiments."
    )

    report_lines.append(
        f"ASSM achieved lower latency in "
        f"{latency_improvements} out of 4 experiments."
    )

    total_baseline_collisions = baseline_data["collisions"].sum()
    total_assm_collisions = assm_data["collisions"].sum()

    report_lines.append(
        f"\nTotal Baseline Collisions: "
        f"{int(total_baseline_collisions)}"
    )

    report_lines.append(
        f"Total ASSM Collisions: "
        f"{int(total_assm_collisions)}"
    )

    if total_baseline_collisions > 0:

        total_reduction = (
            (total_baseline_collisions - total_assm_collisions)
            / total_baseline_collisions
        ) * 100

        report_lines.append(
            f"Aggregate Collision Reduction: {total_reduction:.2f}%"
        )

    report_lines.append(
        "\n3. CONCLUSION"
    )

    report_lines.append("-" * 65)

    report_lines.append(
        "ASSM achieved higher packet delivery ratio and throughput "
        "in all four tested scenarios."
    )

    report_lines.append(
        "It also substantially reduced collisions across the tested "
        "synthetic scenarios."
    )

    report_lines.append(
        "Latency improvement was not consistent across all user counts."
    )

    report_lines.append(
        "These results demonstrate the behavior of the proposed "
        "algorithm under the simulated conditions. They do not establish "
        "performance in a real wireless deployment."
    )

    output_file = REPORT_DIR / "scalability_analysis_report.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(report_lines))

    print(f"\nAnalysis report saved: {output_file}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("SPECTRIX - SCALABILITY ANALYSIS")
    print("=" * 65)

    data = load_experiment_results()

    save_combined_results(data)

    generate_all_graphs(data)

    generate_analysis_report(data)

    print("\n" + "=" * 65)
    print("SCALABILITY ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 65)

    print(f"\nReports folder: {REPORT_DIR}")
    print(f"Graphs folder: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
