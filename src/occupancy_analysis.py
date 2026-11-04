
"""
SPECTRIX - Detailed Occupancy Analysis
Part A: Database Analysis and Visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ==========================================
# 1. PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "raw" / "occupancy_7days.csv"

GRAPH_DIR = BASE_DIR / "outputs" / "graphs"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATABASE
# ==========================================

def load_database():

    print("\nLoading occupancy database...")

    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Database loaded successfully!")
    print(f"Records: {len(df):,}")

    return df


# ==========================================
# 3. CHANNEL-WISE OCCUPANCY
# ==========================================

def channel_statistics(df):

    summary = df.groupby(
        ["channel_id", "channel_type"]
    ).agg(
        total_slots=("busy_state", "count"),
        busy_slots=("busy_state", "sum"),
        average_occupancy=("busy_state", "mean")
    )

    summary["busy_percentage"] = (
        summary["average_occupancy"] * 100
    )

    summary["idle_percentage"] = (
        100 - summary["busy_percentage"]
    )

    summary.to_csv(
        REPORT_DIR / "channel_statistics.csv"
    )

    print("\nCHANNEL-WISE STATISTICS")
    print(summary.round(2).to_string())

    return summary


# ==========================================
# 4. HOURLY OCCUPANCY ANALYSIS
# ==========================================

def hourly_analysis(df):

    hourly = df.pivot_table(
        index="channel_id",
        columns="hour",
        values="busy_state",
        aggfunc="mean"
    ) * 100

    hourly.to_csv(
        REPORT_DIR / "hourly_occupancy.csv"
    )

    # Line graph
    plt.figure(figsize=(14, 7))

    for channel in hourly.index:

        plt.plot(
            hourly.columns,
            hourly.loc[channel],
            marker="o",
            markersize=3,
            label=channel
        )

    plt.title("24-Hour Wireless Channel Occupancy")
    plt.xlabel("Hour of Day")
    plt.ylabel("Busy Percentage (%)")

    plt.xticks(range(24))
    plt.ylim(0, 105)

    plt.grid(alpha=0.3)
    plt.legend(ncol=4)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "hourly_occupancy.png",
        dpi=300
    )

    plt.close()

    # Heatmap
    plt.figure(figsize=(15, 5))

    plt.imshow(
        hourly.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=100
    )

    plt.colorbar(label="Occupancy (%)")

    plt.xticks(range(24), range(24))
    plt.yticks(
        range(len(hourly.index)),
        hourly.index
    )

    plt.xlabel("Hour of Day")
    plt.ylabel("Wireless Channel")
    plt.title("24-Hour Channel Occupancy Heatmap")

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "channel_heatmap.png",
        dpi=300
    )

    plt.close()

    print("\nHourly occupancy analysis completed.")


# ==========================================
# 5. TIME-PERIOD ANALYSIS
# ==========================================

def time_period_analysis(df):

    period_order = [
        "Midnight",
        "Early Morning",
        "Morning",
        "Afternoon",
        "Evening",
        "Late Night"
    ]

    period_summary = df.pivot_table(
        index="channel_id",
        columns="time_period",
        values="busy_state",
        aggfunc="mean"
    ) * 100

    period_summary = period_summary.reindex(
        columns=period_order
    )

    period_summary.to_csv(
        REPORT_DIR / "time_period_occupancy.csv"
    )

    print("\nTIME-PERIOD OCCUPANCY (%)")
    print(period_summary.round(2).to_string())

    ax = period_summary.plot(
        kind="bar",
        figsize=(14, 7),
        width=0.8
    )

    ax.set_title("Channel Occupancy Across Daily Time Periods")
    ax.set_xlabel("Channel ID")
    ax.set_ylabel("Busy Percentage (%)")

    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.legend(title="Time Period")

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "time_period_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 6. SEVEN-DAY OCCUPANCY ANALYSIS
# ==========================================

def weekly_analysis(df):

    days = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]

    weekly = df.pivot_table(
        index="weekday",
        columns="channel_id",
        values="busy_state",
        aggfunc="mean"
    ) * 100

    weekly = weekly.reindex(days)

    weekly.to_csv(
        REPORT_DIR / "weekly_occupancy.csv"
    )

    print("\nSEVEN-DAY CHANNEL OCCUPANCY (%)")
    print(weekly.round(2).to_string())

    plt.figure(figsize=(13, 6))

    weekly.plot(
        marker="o",
        figsize=(13, 6)
    )

    plt.title("Weekly Wireless Channel Occupancy")
    plt.xlabel("Day of Week")
    plt.ylabel("Busy Percentage (%)")

    plt.ylim(0, 105)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "weekly_occupancy.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 7. WEEKEND EFFECT ANALYSIS
# ==========================================

def weekend_analysis(df):

    weekend = df.groupby(
        ["channel_id", "is_weekend"]
    )["busy_state"].mean().unstack() * 100

    weekend = weekend.rename(
        columns={
            False: "Weekday (%)",
            True: "Weekend (%)"
        }
    )

    weekend.to_csv(
        REPORT_DIR / "weekend_analysis.csv"
    )

    print("\nWEEKDAY VS WEEKEND OCCUPANCY")
    print(weekend.round(2).to_string())

    ax = weekend.plot(
        kind="bar",
        figsize=(11, 6)
    )

    ax.set_title("Weekday vs Weekend Channel Occupancy")
    ax.set_xlabel("Channel ID")
    ax.set_ylabel("Busy Percentage (%)")

    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        GRAPH_DIR / "weekend_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\n===================================")
    print(" SPECTRIX OCCUPANCY ANALYSIS")
    print("===================================")

    df = load_database()

    channel_statistics(df)

    hourly_analysis(df)

    time_period_analysis(df)

    weekly_analysis(df)

    weekend_analysis(df)

    print("\n===================================")
    print(" ANALYSIS COMPLETED SUCCESSFULLY")
    print("===================================")

    print(f"\nGraphs saved at: {GRAPH_DIR}")

    print(f"Reports saved at: {REPORT_DIR}")
