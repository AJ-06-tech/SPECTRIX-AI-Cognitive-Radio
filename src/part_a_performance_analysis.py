
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# SPECTRIX PART A - PERFORMANCE ANALYSIS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "outputs" / "reports"
GRAPH_DIR = BASE_DIR / "outputs" / "graphs"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SIMULATION REPORTS
# ============================================================

def load_reports():

    print("\nSTEP 1: LOADING SIMULATION REPORTS")

    required_files = {
        "slots": "part_a_slot_results.csv",
        "overall": "part_a_overall_metrics.csv",
        "period": "part_a_time_period_metrics.csv",
        "users": "part_a_user_metrics.csv"
    }

    data = {}

    for key, filename in required_files.items():

        file_path = REPORT_DIR / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"Required report not found: {file_path}"
            )

        data[key] = pd.read_csv(file_path)

        print(
            f"{filename} loaded successfully | "
            f"Records: {len(data[key])}"
        )

    return data


# ============================================================
# CALCULATE JAIN'S FAIRNESS INDEX
# ============================================================

def calculate_jain_index(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0 or np.sum(values ** 2) == 0:
        return 0.0

    fairness = (
        np.sum(values) ** 2
        /
        (len(values) * np.sum(values ** 2))
    )

    return float(fairness)


# ============================================================
# GRAPH 1 - DAILY PACKET DELIVERY
# ============================================================

def plot_daily_delivery(slots):

    print("\nGRAPH 1: DAILY PACKET DELIVERY")

    slots = slots.copy()

    slots["simulation_time"] = pd.to_datetime(
        slots["simulation_time"]
    )

    slots["day"] = slots["simulation_time"].dt.date

    daily = slots.groupby("day").agg(

        requests=("requests_generated", "sum"),

        delivered=("successful_packets", "sum")

    ).reset_index()

    daily["delivery_ratio"] = np.where(

        daily["requests"] > 0,

        daily["delivered"] / daily["requests"] * 100,

        0
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        daily["day"],
        daily["delivery_ratio"],
        marker="o",
        linewidth=2,
        label="Packet Delivery Ratio"
    )

    plt.axhline(
        daily["delivery_ratio"].mean(),
        linestyle="--",
        label="Seven-day average"
    )

    plt.title("SPECTRIX - Daily Packet Delivery Ratio")

    plt.xlabel("Simulation Day")

    plt.ylabel("Packet Delivery Ratio (%)")

    plt.ylim(0, 105)

    plt.xticks(rotation=30)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_daily_delivery.png",
        dpi=300
    )

    plt.close()

    daily.to_csv(
        REPORT_DIR / "part_a_daily_delivery.csv",
        index=False
    )

    print("Daily delivery graph saved.")


# ============================================================
# GRAPH 2 - DAILY THROUGHPUT
# ============================================================

def plot_daily_throughput(slots):

    print("\nGRAPH 2: DAILY THROUGHPUT")

    slots = slots.copy()

    slots["simulation_time"] = pd.to_datetime(
        slots["simulation_time"]
    )

    slots["day"] = slots["simulation_time"].dt.date

    daily = slots.groupby("day").agg(

        bytes_transmitted=("bytes_transmitted", "sum"),

        slots=("simulation_time", "count")

    ).reset_index()

    daily["throughput_bps"] = (

        daily["bytes_transmitted"] * 8

        /

        (daily["slots"] * 10)
    )

    plt.figure(figsize=(11, 6))

    plt.bar(
        daily["day"].astype(str),
        daily["throughput_bps"]
    )

    plt.title("SPECTRIX - Daily Average Throughput")

    plt.xlabel("Simulation Day")

    plt.ylabel("Throughput (bps)")

    plt.xticks(rotation=30)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_daily_throughput.png",
        dpi=300
    )

    plt.close()

    daily.to_csv(
        REPORT_DIR / "part_a_daily_throughput.csv",
        index=False
    )

    print("Daily throughput graph saved.")


# ============================================================
# GRAPH 3 - TRANSMISSION PERFORMANCE
# ============================================================

def plot_transmission_performance(slots):

    print("\nGRAPH 3: TRANSMISSION PERFORMANCE")

    totals = {

        "Successful": slots[
            "successful_packets"
        ].sum(),

        "Failed attempts": (

            slots["transmission_attempts"].sum()

            -

            slots["successful_packets"].sum()
        )
    }

    plt.figure(figsize=(9, 6))

    plt.bar(
        list(totals.keys()),
        list(totals.values())
    )

    plt.title("SPECTRIX - Transmission Performance")

    plt.xlabel("Transmission Outcome")

    plt.ylabel("Number of Attempts")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_transmission_performance.png",
        dpi=300
    )

    plt.close()

    print("Transmission performance graph saved.")


# ============================================================
# GRAPH 4 - TIME-PERIOD PERFORMANCE
# ============================================================

def plot_time_period_performance(period):

    print("\nGRAPH 4: TIME-PERIOD PERFORMANCE")

    period_order = [

        "Midnight",

        "Early Morning",

        "Morning",

        "Afternoon",

        "Evening",

        "Late Night"
    ]

    period = period.copy()

    period["delivery_ratio"] = np.where(

        period["requests_generated"] > 0,

        period["successful_packets"]
        / period["requests_generated"] * 100,

        0
    )

    period["throughput_bps"] = (

        period["bytes_transmitted"] * 8

        /

        (period["slots"] * 10)
    )

    period["collision_rate"] = np.where(

        period["attempts"] > 0,

        period["collisions"] / period["attempts"] * 100,

        0
    )

    period["time_period"] = pd.Categorical(

        period["time_period"],

        categories=period_order,

        ordered=True
    )

    period = period.sort_values(
        "time_period"
    )

    plt.figure(figsize=(12, 6))

    plt.bar(

        period["time_period"].astype(str),

        period["delivery_ratio"]
    )

    plt.title(
        "SPECTRIX - Packet Delivery Across Time Periods"
    )

    plt.xlabel("Time Period")

    plt.ylabel("Packet Delivery Ratio (%)")

    plt.xticks(rotation=25)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_period_delivery.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(12, 6))

    plt.plot(

        period["time_period"].astype(str),

        period["throughput_bps"],

        marker="o",

        linewidth=2
    )

    plt.title(
        "SPECTRIX - Throughput Across Time Periods"
    )

    plt.xlabel("Time Period")

    plt.ylabel("Throughput (bps)")

    plt.xticks(rotation=25)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_period_throughput.png",
        dpi=300
    )

    plt.close()

    period.to_csv(
        REPORT_DIR / "part_a_period_detailed_analysis.csv",
        index=False
    )

    print("Time-period graphs saved.")


# ============================================================
# GRAPH 5 - SECONDARY USER PERFORMANCE
# ============================================================

def plot_user_performance(users):

    print("\nGRAPH 5: SECONDARY USER PERFORMANCE")

    users = users.copy()

    users["delivery_ratio"] = np.where(

        users["total_requests"] > 0,

        users["total_packets_sent"]
        / users["total_requests"] * 100,

        0
    )

    plt.figure(figsize=(10, 6))

    plt.bar(

        users["user_id"],

        users["total_packets_sent"]
    )

    plt.title(
        "SPECTRIX - Successful Packets per Secondary User"
    )

    plt.xlabel("Secondary User")

    plt.ylabel("Successfully Delivered Packets")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_user_packets.png",
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(10, 6))

    plt.bar(

        users["user_id"],

        users["average_waiting_time"]
    )

    plt.title(
        "SPECTRIX - Average Packet Waiting Time per User"
    )

    plt.xlabel("Secondary User")

    plt.ylabel("Average Waiting Time (seconds)")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_user_latency.png",
        dpi=300
    )

    plt.close()

    print("Secondary user graphs saved.")

    return users


# ============================================================
# GRAPH 6 - FAIRNESS ANALYSIS
# ============================================================

def fairness_analysis(users):

    print("\nGRAPH 6: JAIN'S FAIRNESS ANALYSIS")

    packet_fairness = calculate_jain_index(

        users["total_packets_sent"].values
    )

    delivery_rate_fairness = calculate_jain_index(

        users["delivery_ratio"].values
    )

    waiting_times = users[
        "average_waiting_time"
    ].values

    waiting_time_fairness = calculate_jain_index(

        waiting_times
    )

    fairness_results = {

        "packet_delivery_fairness":
            round(packet_fairness, 5),

        "delivery_rate_fairness":
            round(delivery_rate_fairness, 5),

        "waiting_time_fairness":
            round(waiting_time_fairness, 5)
    }

    plt.figure(figsize=(9, 6))

    plt.bar(

        list(fairness_results.keys()),

        list(fairness_results.values())
    )

    plt.title("SPECTRIX - Jain's Fairness Indices")

    plt.ylabel("Fairness Index (0 to 1)")

    plt.ylim(0, 1.05)

    plt.xticks(rotation=20)

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_fairness_analysis.png",
        dpi=300
    )

    plt.close()

    pd.DataFrame(
        [fairness_results]
    ).to_csv(
        REPORT_DIR / "part_a_fairness_metrics.csv",
        index=False
    )

    print("\nFairness results:")

    for metric, value in fairness_results.items():

        print(metric, ":", value)

    return fairness_results


# ============================================================
# GRAPH 7 - COLLISION AND HANDOVER ANALYSIS
# ============================================================

def plot_collision_handover(overall):

    print("\nGRAPH 7: COLLISION AND HANDOVER ANALYSIS")

    labels = [
        "Collisions",
        "Handovers"
    ]

    values = [

        overall["total_collisions"],

        overall["total_handovers"]
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(labels, values)

    plt.title(
        "SPECTRIX - Collision and Handover Counts"
    )

    plt.xlabel("Performance Metric")

    plt.ylabel("Count")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "part_a_collision_handover.png",
        dpi=300
    )

    plt.close()

    print("Collision and handover graph saved.")


# ============================================================
# SAVE FINAL PERFORMANCE SUMMARY
# ============================================================

def save_performance_summary(overall, users, fairness):

    print("\nSAVING FINAL PERFORMANCE SUMMARY")

    summary = {

        "simulation_slots":
            overall["simulation_slots"],

        "simulation_days":
            overall["simulation_days"],

        "packet_delivery_ratio_percent":
            overall["packet_delivery_ratio_percent"],

        "throughput_bps":
            overall["throughput_bps"],

        "average_latency_seconds":
            overall["average_packet_latency_seconds"],

        "collision_rate_percent":
            overall["collision_rate_percent"],

        "total_handovers":
            overall["total_handovers"],

        "total_packets_delivered":
            overall["successfully_delivered_packets"],

        "total_failed_packets":
            overall["failed_packets"],

        "packet_delivery_fairness":
            fairness["packet_delivery_fairness"],

        "delivery_rate_fairness":
            fairness["delivery_rate_fairness"],

        "waiting_time_fairness":
            fairness["waiting_time_fairness"]
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        REPORT_DIR / "part_a_final_performance_summary.csv",
        index=False
    )

    print("\nFinal performance summary:")

    for metric, value in summary.items():

        print(f"{metric}: {value}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(" SPECTRIX - PART A PERFORMANCE ANALYSIS")

    print("=" * 70)

    data = load_reports()

    slots = data["slots"]

    overall = data["overall"].iloc[0].to_dict()

    period = data["period"]

    users = data["users"]

    # --------------------------------------------------------
    # GENERATE ALL GRAPHS
    # --------------------------------------------------------

    plot_daily_delivery(slots)

    plot_daily_throughput(slots)

    plot_transmission_performance(slots)

    plot_time_period_performance(period)

    users = plot_user_performance(users)

    fairness = fairness_analysis(users)

    plot_collision_handover(overall)

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    save_performance_summary(
        overall,
        users,
        fairness
    )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(" PART A PERFORMANCE ANALYSIS COMPLETED")

    print("=" * 70)

    print("\nGraphs saved in:")
    print(GRAPH_DIR)

    print("\nReports saved in:")
    print(REPORT_DIR)


if __name__ == "__main__":

    main()
