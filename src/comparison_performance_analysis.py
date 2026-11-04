
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# SPECTRIX - COMPARATIVE PERFORMANCE ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = (
    BASE_DIR
    / "outputs"
    / "reports"
    / "baseline_comparison"
)

GRAPH_DIR = (
    BASE_DIR
    / "outputs"
    / "graphs"
    / "baseline_comparison"
)

GRAPH_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD COMPARISON REPORTS
# ============================================================

def load_reports():

    print("\nSTEP 1: LOADING COMPARISON REPORTS")

    files = {

        "overall": (
            REPORT_DIR / "algorithm_comparison.csv"
        ),

        "baseline_users": (
            REPORT_DIR / "BASELINE" / "user_metrics.csv"
        ),

        "assm_users": (
            REPORT_DIR / "ASSM" / "user_metrics.csv"
        ),

        "baseline_channels": (
            REPORT_DIR / "BASELINE"
            / "channel_allocation_metrics.csv"
        ),

        "assm_channels": (
            REPORT_DIR / "ASSM"
            / "channel_allocation_metrics.csv"
        ),

        "baseline_period": (
            REPORT_DIR / "BASELINE"
            / "time_period_metrics.csv"
        ),

        "assm_period": (
            REPORT_DIR / "ASSM"
            / "time_period_metrics.csv"
        )

    }

    data = {}

    for name, path in files.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing report: {path}"
            )

        data[name] = pd.read_csv(path)

        print(
            f"{name}: loaded "
            f"({len(data[name])} records)"
        )

    return data


# ============================================================
# VALIDATE REPORTS
# ============================================================

def validate_reports(data):

    print("\nSTEP 2: VALIDATING REPORTS")

    overall = data["overall"]

    if len(overall) != 2:

        raise ValueError(
            "Expected exactly two algorithms."
        )

    expected_algorithms = {
        "BASELINE",
        "ASSM"
    }

    actual_algorithms = set(
        overall["algorithm"]
    )

    if actual_algorithms != expected_algorithms:

        raise ValueError(
            "Algorithm names do not match."
        )

    if overall["evaluation_slots"].nunique() != 1:

        raise ValueError(
            "Algorithms do not have the same evaluation window."
        )

    if int(overall["evaluation_slots"].iloc[0]) != 17280:

        raise ValueError(
            "Expected 17280 held-out evaluation slots."
        )

    print("Two algorithms found: PASSED")

    print("Same evaluation window: PASSED")

    print("Held-out slot count: PASSED")


# ============================================================
# HELPER: GET ALGORITHM VALUE
# ============================================================

def get_metric(overall, algorithm, metric):

    result = overall.loc[
        overall["algorithm"] == algorithm,
        metric
    ]

    if result.empty:

        raise ValueError(
            f"Metric {metric} missing for {algorithm}"
        )

    return float(result.iloc[0])


# ============================================================
# GRAPH 1: OVERALL PERFORMANCE COMPARISON
# ============================================================

def plot_overall_performance(overall):

    print("\nGRAPH 1: OVERALL PERFORMANCE")

    metrics = [

        (
            "packet_delivery_ratio_percent",
            "Packet Delivery Ratio (%)"
        ),

        (
            "throughput_bps",
            "Throughput (bps)"
        ),

        (
            "average_latency_seconds",
            "Average Latency (seconds)"
        ),

        (
            "collision_rate_percent",
            "Collision Rate (%)"
        )

    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 9)
    )

    for axis, (column, title) in zip(
        axes.flatten(),
        metrics
    ):

        baseline = get_metric(
            overall,
            "BASELINE",
            column
        )

        assm = get_metric(
            overall,
            "ASSM",
            column
        )

        axis.bar(
            ["Baseline", "ASSM"],
            [baseline, assm]
        )

        axis.set_title(title)

        axis.set_ylabel(title)

        axis.grid(
            axis="y",
            alpha=0.3
        )

        for index, value in enumerate(
            [baseline, assm]
        ):

            axis.text(
                index,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

    fig.suptitle(
        "SPECTRIX - Baseline vs ASSM Performance",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "overall_performance_comparison.png",
        dpi=300
    )

    plt.close()

    print("Overall performance graph saved.")


# ============================================================
# GRAPH 2: COLLISION RATE
# ============================================================

def plot_collision_comparison(overall):

    print("\nGRAPH 2: COLLISION RATE")

    baseline = get_metric(
        overall,
        "BASELINE",
        "collision_rate_percent"
    )

    assm = get_metric(
        overall,
        "ASSM",
        "collision_rate_percent"
    )

    reduction = (

        (baseline - assm) / baseline * 100

        if baseline > 0

        else 0

    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        ["Standard Baseline", "ASSM"],
        [baseline, assm]
    )

    plt.title(
        "Collision Rate Comparison"
    )

    plt.ylabel("Collision Rate (%)")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "collision_rate_comparison.png",
        dpi=300
    )

    plt.close()

    print(
        f"Relative collision reduction: {reduction:.4f}%"
    )


# ============================================================
# GRAPH 3: CHANNEL UTILIZATION
# ============================================================

def plot_channel_utilization(overall):

    print("\nGRAPH 3: CHANNEL UTILIZATION")

    baseline = get_metric(
        overall,
        "BASELINE",
        "channel_utilization_percent"
    )

    assm = get_metric(
        overall,
        "ASSM",
        "channel_utilization_percent"
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        ["Standard Baseline", "ASSM"],
        [baseline, assm]
    )

    plt.title(
        "Channel Allocation Utilization"
    )

    plt.ylabel(
        "Allocated Transmissions / Channel-Slot Capacity (%)"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "channel_utilization_comparison.png",
        dpi=300
    )

    plt.close()

    print("Channel utilization graph saved.")


# ============================================================
# GRAPH 4: CHANNEL ALLOCATION DISTRIBUTION
# ============================================================

def plot_channel_allocations(
    baseline_channels,
    assm_channels
):

    print("\nGRAPH 4: CHANNEL ALLOCATION DISTRIBUTION")

    merged = baseline_channels.merge(

        assm_channels,

        on="channel_id",

        suffixes=("_baseline", "_assm")

    )

    x = np.arange(
        len(merged)
    )

    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(

        x - width / 2,

        merged["allocation_count_baseline"],

        width,

        label="Baseline"

    )

    plt.bar(

        x + width / 2,

        merged["allocation_count_assm"],

        width,

        label="ASSM"

    )

    plt.xticks(
        x,
        merged["channel_id"]
    )

    plt.xlabel("Channel")

    plt.ylabel("Number of Allocations")

    plt.title(
        "Channel Allocation Distribution"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "channel_allocation_comparison.png",
        dpi=300
    )

    plt.close()

    print("Channel allocation graph saved.")


# ============================================================
# GRAPH 5: SECONDARY USER PERFORMANCE
# ============================================================

def plot_user_performance(
    baseline_users,
    assm_users
):

    print("\nGRAPH 5: SECONDARY USER PERFORMANCE")

    merged = baseline_users.merge(

        assm_users,

        on="user_id",

        suffixes=("_baseline", "_assm")

    )

    x = np.arange(
        len(merged)
    )

    width = 0.35

    # --------------------------------------------------------
    # DELIVERED PACKETS
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.bar(

        x - width / 2,

        merged["packets_delivered_baseline"],

        width,

        label="Baseline"

    )

    plt.bar(

        x + width / 2,

        merged["packets_delivered_assm"],

        width,

        label="ASSM"

    )

    plt.xticks(
        x,
        merged["user_id"]
    )

    plt.xlabel("Secondary User")

    plt.ylabel("Successfully Delivered Packets")

    plt.title(
        "Secondary User Packet Delivery Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "user_delivery_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # AVERAGE LATENCY
    # --------------------------------------------------------

    plt.figure(figsize=(11, 6))

    plt.bar(

        x - width / 2,

        merged["average_latency_seconds_baseline"],

        width,

        label="Baseline"

    )

    plt.bar(

        x + width / 2,

        merged["average_latency_seconds_assm"],

        width,

        label="ASSM"

    )

    plt.xticks(
        x,
        merged["user_id"]
    )

    plt.xlabel("Secondary User")

    plt.ylabel("Average Latency (seconds)")

    plt.title(
        "Secondary User Latency Comparison"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "user_latency_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # SAVE MERGED USER REPORT
    # --------------------------------------------------------

    merged.to_csv(
        REPORT_DIR / "user_comparison.csv",
        index=False
    )

    print("Secondary user graphs saved.")

    return merged


# ============================================================
# GRAPH 6: SIX TIME-PERIOD COMPARISON
# ============================================================

def plot_time_period_comparison(
    baseline_period,
    assm_period
):

    print("\nGRAPH 6: TIME-PERIOD COMPARISON")

    merged = baseline_period.merge(

        assm_period,

        on="time_period",

        suffixes=("_baseline", "_assm")

    )

    period_order = [

        "Midnight",

        "Early Morning",

        "Morning",

        "Afternoon",

        "Evening",

        "Late Night"

    ]

    merged["time_period"] = pd.Categorical(

        merged["time_period"],

        categories=period_order,

        ordered=True

    )

    merged = merged.sort_values(
        "time_period"
    )

    # --------------------------------------------------------
    # CALCULATE PERIOD THROUGHPUT
    # --------------------------------------------------------

    merged["throughput_baseline"] = (

        merged["bytes_transmitted_baseline"] * 8

        /

        (merged["slots_baseline"] * 10)

    )

    merged["throughput_assm"] = (

        merged["bytes_transmitted_assm"] * 8

        /

        (merged["slots_assm"] * 10)

    )

    # --------------------------------------------------------
    # CALCULATE PERIOD COLLISION RATE
    # --------------------------------------------------------

    merged["collision_rate_baseline"] = np.where(

        merged["attempts_baseline"] > 0,

        merged["collisions_baseline"]
        / merged["attempts_baseline"] * 100,

        0

    )

    merged["collision_rate_assm"] = np.where(

        merged["attempts_assm"] > 0,

        merged["collisions_assm"]
        / merged["attempts_assm"] * 100,

        0

    )

    x = np.arange(
        len(merged)
    )

    width = 0.35

    # --------------------------------------------------------
    # PERIOD THROUGHPUT GRAPH
    # --------------------------------------------------------

    plt.figure(figsize=(13, 6))

    plt.bar(

        x - width / 2,

        merged["throughput_baseline"],

        width,

        label="Baseline"

    )

    plt.bar(

        x + width / 2,

        merged["throughput_assm"],

        width,

        label="ASSM"

    )

    plt.xticks(
        x,
        merged["time_period"].astype(str),
        rotation=20
    )

    plt.xlabel("Time Period")

    plt.ylabel("Throughput (bps)")

    plt.title(
        "Throughput Across Six Time Periods"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "period_throughput_comparison.png",
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # PERIOD COLLISION GRAPH
    # --------------------------------------------------------

    plt.figure(figsize=(13, 6))

    plt.bar(

        x - width / 2,

        merged["collision_rate_baseline"],

        width,

        label="Baseline"

    )

    plt.bar(

        x + width / 2,

        merged["collision_rate_assm"],

        width,

        label="ASSM"

    )

    plt.xticks(
        x,
        merged["time_period"].astype(str),
        rotation=20
    )

    plt.xlabel("Time Period")

    plt.ylabel("Collision Rate (%)")

    plt.title(
        "Collision Rate Across Six Time Periods"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "period_collision_comparison.png",
        dpi=300
    )

    plt.close()

    merged.to_csv(
        REPORT_DIR / "period_comparison.csv",
        index=False
    )

    print("Time-period graphs saved.")


# ============================================================
# GRAPH 7: FAIRNESS COMPARISON
# ============================================================

def plot_fairness_comparison(overall):

    print("\nGRAPH 7: FAIRNESS COMPARISON")

    baseline = get_metric(

        overall,

        "BASELINE",

        "jain_delivery_fairness"

    )

    assm = get_metric(

        overall,

        "ASSM",

        "jain_delivery_fairness"

    )

    plt.figure(figsize=(9, 6))

    plt.bar(

        ["Baseline", "ASSM"],

        [baseline, assm]

    )

    plt.title(
        "Jain's Packet Delivery Fairness Comparison"
    )

    plt.ylabel("Jain's Fairness Index")

    plt.ylim(0, 1.05)

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "fairness_comparison.png",
        dpi=300
    )

    plt.close()

    print(
        "Baseline fairness:",
        round(baseline, 5)
    )

    print(
        "ASSM fairness:",
        round(assm, 5)
    )


# ============================================================
# CALCULATE IMPROVEMENTS
# ============================================================

def calculate_improvements(overall):

    print("\nCALCULATING PERFORMANCE IMPROVEMENTS")

    metric_directions = {

        "packet_delivery_ratio_percent": "higher",

        "throughput_bps": "higher",

        "average_latency_seconds": "lower",

        "collision_rate_percent": "lower",

        "channel_utilization_percent": "higher",

        "jain_delivery_fairness": "higher"

    }

    results = []

    for metric, direction in metric_directions.items():

        baseline = get_metric(
            overall,
            "BASELINE",
            metric
        )

        assm = get_metric(
            overall,
            "ASSM",
            metric
        )

        absolute_difference = assm - baseline

        if baseline != 0:

            relative_change = (
                absolute_difference / baseline * 100
            )

        else:

            relative_change = 0

        if direction == "higher":

            improvement = relative_change

        else:

            improvement = -relative_change

        results.append({

            "metric": metric,

            "baseline": baseline,

            "assm": assm,

            "absolute_difference": absolute_difference,

            "relative_change_percent": relative_change,

            "improvement_direction": direction,

            "assm_improvement_percent": improvement

        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(

        REPORT_DIR / "performance_improvements.csv",

        index=False

    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)

    print(" SPECTRIX - COMPARATIVE PERFORMANCE ANALYSIS")

    print("=" * 75)

    data = load_reports()

    validate_reports(data)

    overall = data["overall"]

    # --------------------------------------------------------
    # GENERATE ALL GRAPHS
    # --------------------------------------------------------

    plot_overall_performance(overall)

    plot_collision_comparison(overall)

    plot_channel_utilization(overall)

    plot_channel_allocations(

        data["baseline_channels"],

        data["assm_channels"]

    )

    plot_user_performance(

        data["baseline_users"],

        data["assm_users"]

    )

    plot_time_period_comparison(

        data["baseline_period"],

        data["assm_period"]

    )

    plot_fairness_comparison(overall)

    # --------------------------------------------------------
    # PERFORMANCE IMPROVEMENTS
    # --------------------------------------------------------

    improvements = calculate_improvements(
        overall
    )

    print("\n" + "=" * 75)

    print(" PERFORMANCE IMPROVEMENT SUMMARY")

    print("=" * 75)

    print(
        improvements.to_string(index=False)
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 75)

    print(" COMPARATIVE ANALYSIS COMPLETED")

    print("=" * 75)

    print("\nGraphs saved in:")

    print(GRAPH_DIR)

    print("\nReports saved in:")

    print(REPORT_DIR)

    print("\nGenerated graphs:")

    print("1. overall_performance_comparison.png")

    print("2. collision_rate_comparison.png")

    print("3. channel_utilization_comparison.png")

    print("4. channel_allocation_comparison.png")

    print("5. user_delivery_comparison.png")

    print("6. user_latency_comparison.png")

    print("7. period_throughput_comparison.png")

    print("8. period_collision_comparison.png")

    print("9. fairness_comparison.png")


if __name__ == "__main__":

    main()
