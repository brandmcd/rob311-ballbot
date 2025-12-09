#!/usr/bin/env python3
"""
IMU Drift Test - Baseline (Motors OFF)
========================================
Tests IMU drift when robot is stationary on a flat surface.

Usage:
  1. Power-cycle the Pico
  2. Wait for IMU initialization (~5 seconds)
  3. Place robot on FLAT, LEVEL surface
  4. DO NOT TOUCH the robot
  5. Run: python3 test_imu_drift_baseline.py
  6. Choose duration: 30s, 60s, or 180s

Output:
  - Saves data to: imu_drift_baseline_XXs.txt
  - Prints real-time drift statistics
"""

import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger3 import dataLogger

# Global state
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0

def feedback_handler(channel, data):
    global msg, last_time
    last_time = time.time()
    msg = mbot_balbot_feedback_t.decode(data)

def lcm_listener(lc):
    global listening
    while listening:
        try:
            lc.handle_timeout(100)
        except Exception as e:
            print(f"LCM error: {e}")
            break

def main():
    global listening, msg

    # Setup
    print("=" * 60)
    print("IMU DRIFT TEST - BASELINE (Motors OFF)")
    print("=" * 60)
    print("\nIMPORTANT:")
    print("  1. Power-cycle the Pico and wait 5 seconds")
    print("  2. Place robot on FLAT, LEVEL surface")
    print("  3. DO NOT TOUCH the robot during test")
    print()

    duration_choice = input("Choose test duration (1=30s, 2=60s, 3=180s): ")
    durations = {"1": 30, "2": 60, "3": 180}
    duration = durations.get(duration_choice, 30)

    filename = f"imu_drift_baseline_{duration}s.txt"
    dl = dataLogger(filename)

    # LCM setup
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()

    print(f"\nStarting {duration}s test...")
    print("Waiting for first message...")
    time.sleep(0.5)

    # Zero angles at start
    theta_x_0 = msg.imu_angles_rpy[0]
    theta_y_0 = msg.imu_angles_rpy[1]
    theta_z_0 = msg.imu_angles_rpy[2]

    print(f"Initial angles: roll={np.rad2deg(theta_x_0):.3f}°, pitch={np.rad2deg(theta_y_0):.3f}°, yaw={np.rad2deg(theta_z_0):.3f}°")
    print("\nLogging data... DO NOT TOUCH ROBOT!\n")

    # Data logging header
    header = ["i", "t", "roll_deg", "pitch_deg", "yaw_deg"]
    dl.appendData(header)

    # Log data
    i = 0
    t_start = time.time()

    try:
        while True:
            time.sleep(0.01)  # 100 Hz sampling
            t_now = time.time() - t_start

            if t_now > duration:
                break

            # Read IMU (zeroed)
            roll = msg.imu_angles_rpy[0] - theta_x_0
            pitch = msg.imu_angles_rpy[1] - theta_y_0
            yaw = msg.imu_angles_rpy[2] - theta_z_0

            # Log data
            data = [
                i, t_now,
                np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)
            ]
            dl.appendData(data)

            # Print status every 5 seconds
            if i % 500 == 0:
                print(f"t={t_now:6.1f}s | roll={np.rad2deg(roll):+6.3f}° | pitch={np.rad2deg(pitch):+6.3f}° | yaw={np.rad2deg(yaw):+6.3f}°")

            i += 1

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print(f"\nTest complete! Logged {i} samples over {t_now:.1f}s")
        print(f"Saving data to {filename}...")
        dl.writeOut()
        listening = False
        listener_thread.join(timeout=1)

        # Quick analysis (only if we collected data)
        if i > 0:
            print("\n" + "=" * 60)
            print("QUICK ANALYSIS")
            print("=" * 60)

            # Read back the data
            with open(filename, 'r') as f:
                lines = f.readlines()[1:]  # Skip header

            if len(lines) > 0:
                rolls = [float(line.split()[2]) for line in lines if len(line.split()) > 3]
                pitches = [float(line.split()[3]) for line in lines if len(line.split()) > 3]

                if rolls and pitches:
                    print(f"Roll:  min={min(rolls):+6.3f}°, max={max(rolls):+6.3f}°, range={max(rolls)-min(rolls):6.3f}°")
                    print(f"Pitch: min={min(pitches):+6.3f}°, max={max(pitches):+6.3f}°, range={max(pitches)-min(pitches):6.3f}°")
                    print()
                    print("INTERPRETATION:")
                    if abs(max(rolls)-min(rolls)) < 0.5 and abs(max(pitches)-min(pitches)) < 0.5:
                        print("  ✓ Drift is ACCEPTABLE (< 0.5°)")
                    elif abs(max(rolls)-min(rolls)) < 2.0 and abs(max(pitches)-min(pitches)) < 2.0:
                        print("  ⚠ Moderate drift (0.5-2°) - consider offset compensation")
                    else:
                        print("  ✗ Large drift (> 2°) - check IMU orientation and initialization")
                else:
                    print("No valid data collected")
            else:
                print("No data lines found in log file")

        print(f"\nData saved to: {filename}")
        if i > 0:
            print("Plot with: python3 plot_imu_drift.py " + filename)
        else:
            print("WARNING: No samples collected - check if LCM messages are being received")

if __name__ == "__main__":
    main()
