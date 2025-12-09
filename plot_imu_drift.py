#!/usr/bin/env python3
"""
Plot IMU Drift Test Results
============================
Visualizes IMU drift from test logs.

Usage:
  python3 plot_imu_drift.py <logfile>

Example:
  python3 plot_imu_drift.py imu_drift_baseline_30s.txt
"""

import sys
import matplotlib.pyplot as plt

def plot_baseline(filename):
    """Plot baseline drift test (motors off)."""
    # Read data
    times, rolls, pitches, yaws = [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:  # i, t, roll, pitch, yaw
                times.append(float(parts[1]))
                rolls.append(float(parts[2]))
                pitches.append(float(parts[3]))
                yaws.append(float(parts[4]))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Roll and Pitch
    ax1.plot(times, rolls, 'b-', label='Roll', linewidth=1.5)
    ax1.plot(times, pitches, 'r-', label='Pitch', linewidth=1.5)
    ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='±0.5° (acceptable)')
    ax1.axhline(y=-0.5, color='g', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title(f'IMU Drift Test - {filename}')
    ax1.legend()

    # Yaw
    ax2.plot(times, yaws, 'purple', linewidth=1.5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Yaw (degrees)')
    ax2.set_title('Yaw Drift')

    # Stats
    if rolls and pitches:
        roll_range = max(rolls) - min(rolls)
        pitch_range = max(pitches) - min(pitches)
        print(f"\nDrift Analysis for {filename}:")
        print(f"  Roll range:  {roll_range:.3f}°")
        print(f"  Pitch range: {pitch_range:.3f}°")
        if roll_range < 0.5 and pitch_range < 0.5:
            print("  ✓ Drift is ACCEPTABLE (< 0.5°)")
        else:
            print("  ⚠ Drift exceeds 0.5° - consider compensation")
    else:
        print(f"\nNo data found in {filename}")

    plt.tight_layout()
    plt.savefig(filename.replace('.txt', '.png'), dpi=150)
    print(f"\nPlot saved to: {filename.replace('.txt', '.png')}")
    plt.show()

def plot_motors(filename):
    """Plot motor drift test."""
    # Read data
    times, pwms, rolls, pitches = [], [], [], []
    with open(filename, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:  # i, t, pwm, roll, pitch, yaw
                times.append(float(parts[1]))
                pwms.append(float(parts[2]))
                rolls.append(float(parts[3]))
                pitches.append(float(parts[4]))

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Roll and Pitch
    ax1.plot(times, rolls, 'b-', label='Roll', linewidth=1.5)
    ax1.plot(times, pitches, 'r-', label='Pitch', linewidth=1.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('IMU Drift with Motors Running')
    ax1.legend()

    # PWM level
    ax2.plot(times, [p*100 for p in pwms], 'g-', linewidth=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Motor PWM (%)')
    ax2.set_title('Motor Command')

    plt.tight_layout()
    plt.savefig(filename.replace('.txt', '.png'), dpi=150)
    print(f"\nPlot saved to: {filename.replace('.txt', '.png')}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_imu_drift.py <logfile>")
        sys.exit(1)

    filename = sys.argv[1]

    if "motors" in filename:
        plot_motors(filename)
    else:
        plot_baseline(filename)
