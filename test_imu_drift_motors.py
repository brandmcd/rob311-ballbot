#!/usr/bin/env python3
"""
IMU Drift Test - With Motors Running
======================================
Tests IMU drift while motors spin at various PWM levels.
Checks if motor magnetic fields or vibrations affect IMU.

Usage:
  1. SECURE robot on stand or hold safely - wheels must spin freely!
  2. Run: python3 test_imu_drift_motors.py
  3. Motors will spin at: 0%, 25%, 50%, 75% PWM for 15s each

Output:
  - Saves data to: imu_drift_motors.txt
  - Prints real-time drift statistics
"""

import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
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
    print("IMU DRIFT TEST - WITH MOTORS")
    print("=" * 60)
    print("\n⚠ WARNING: SECURE ROBOT ON STAND OR HOLD SAFELY!")
    print("   Wheels must be able to spin freely.\n")

    input("Press Enter when robot is secured...")

    filename = "imu_drift_motors.txt"
    dl = dataLogger(filename)

    # LCM setup
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()

    print("\nStarting test...")
    time.sleep(0.5)

    # Zero angles at start
    theta_x_0 = msg.imu_angles_rpy[0]
    theta_y_0 = msg.imu_angles_rpy[1]
    theta_z_0 = msg.imu_angles_rpy[2]

    print(f"Initial angles: roll={np.rad2deg(theta_x_0):.3f}°, pitch={np.rad2deg(theta_y_0):.3f}°, yaw={np.rad2deg(theta_z_0):.3f}°\n")

    # Data logging header
    header = ["i", "t", "pwm_level", "roll_deg", "pitch_deg", "yaw_deg"]
    dl.appendData(header)
    print("Data logging initialized\n")

    # Test at different PWM levels
    pwm_levels = [0.0, 0.25, 0.5, 0.75]
    duration_per_level = 15  # seconds

    i = 0
    t_start = time.time()
    command = mbot_motor_pwm_t()

    try:
        for pwm in pwm_levels:
            print(f"Testing PWM = {pwm*100:.0f}%...")

            # Set motor PWM
            command.utime = int(time.time() * 1e6)
            command.pwm[0] = pwm
            command.pwm[1] = pwm
            command.pwm[2] = pwm
            lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

            # Log for duration_per_level seconds
            level_start = time.time()
            while time.time() - level_start < duration_per_level:
                time.sleep(0.01)  # 100 Hz
                t_now = time.time() - t_start

                # Read IMU (zeroed)
                roll = msg.imu_angles_rpy[0] - theta_x_0
                pitch = msg.imu_angles_rpy[1] - theta_y_0
                yaw = msg.imu_angles_rpy[2] - theta_z_0

                # Log data
                data = [i, t_now, pwm, np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]
                dl.appendData(data)

                # Print status every 1 second
                if i % 100 == 0:
                    print(f"  t={t_now:5.1f}s | roll={np.rad2deg(roll):+6.3f}° | pitch={np.rad2deg(pitch):+6.3f}°")

                i += 1

            # Stop motors briefly between levels
            command.utime = int(time.time() * 1e6)
            command.pwm[:] = [0.0, 0.0, 0.0]
            lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Stop motors
        print("\nStopping motors...")
        command.utime = int(time.time() * 1e6)
        command.pwm[:] = [0.0, 0.0, 0.0]
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

        print(f"Saving data to {filename}...")
        dl.writeOut()
        listening = False
        listener_thread.join(timeout=1)

        print(f"\nData saved to: {filename}")
        print("Plot with: python3 plot_imu_drift.py " + filename)

if __name__ == "__main__":
    main()
