"""
ROB 311 - Fall 2025
Author: Prof. Greg Formosa
University of Michigan

Script is meant to test MBot Control Board IMU & LCM communications, without additional overhead.
Will store data as a "test_IMU_[#].txt" file that can be parsed in Matlab to show IMU angle data.
Run script, choose a test number to store data as (will overwrite if file already exists), and then
rotate board along multiple axes to take IMU readings. Press Ctrl+C to stop script at any time.

Note: IMU readings are not zeroed at each run; they are zeroed from calibration when Pico was originally booted up.

LCM data type for ROB 311 ball-bots:
struct mbot_balbot_feedback_t
{
    int64_t utime;
    int32_t enc_ticks[3];       // absolute postional ticks
    int32_t enc_delta_ticks[3]; // number of ticks since last step
    int32_t enc_delta_time;     // [usec]
    float imu_angles_rpy[3];    // [radian]
    float volts[4];             // volts
}

"""

import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger import dataLogger

# Constants for the control loop
FREQ = 200  # Frequency of control loop in Hz
DT = 1 / FREQ  # Time step for each iteration in seconds
JOYSTICK_SCALE = 32767  # Scale factor for normalizing joystick values
N_GEARBOX = 70 # Motor gearbox ratio
N_ENC = 64 # Ticks per revolution of encoder

# Global flags to control the listening thread & msg data
listening = False
msg = mbot_balbot_feedback_t()

def feedback_handler(channel, data):
    """Callback function to handle received mbot_balbot_feedback_t messages"""
    global msg
    msg = mbot_balbot_feedback_t.decode(data)

def lcm_listener(lc):
    """Function to continuously listen for LCM messages in a separate thread"""
    global listening
    while listening:
        try:
            lc.handle_timeout(100)  # 100ms timeout
        except Exception as e:
            print(f"LCM listening error: {e}")
            break


def main():
    # === Data Logging Initialization ===
    # Prompt user for trial number and create a data logger
    trial_num = int(input("Test Number? "))
    filename = f"test_IMU_{trial_num}.txt"
    dl = dataLogger(filename)
    
    # === LCM Messaging Initialization ===
    # Initialize the serial communication protocol
    global listening
    global msg
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    # Start a separate thread for reading LCM data
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("Started continuous LCM listener...")

    try:
        command = mbot_motor_pwm_t()
        # === Main Control Loop ===
        print("Starting steering control loop...")
        i = 0  # Iteration counter
        t_start = time.time()
        enc_pos_1_start = msg.enc_ticks[0]
        enc_pos_2_start = msg.enc_ticks[1]
        enc_pos_3_start = msg.enc_ticks[2]
        data = ["i t_now Tx Ty Tz u1 u2 u3 theta_x theta_y theta_z psi_1 psi_2 psi_3 dpsi_1 dpsi_2 dpsi_3"]
        dl.appendData(data)
        Tx = 0
        Ty = 0
        Tz = 0
        u1 = 0
        u2 = 0
        u3 = 0

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start  # Elapsed time
            i += 1

            try:
                # Pull sensor data
                theta_x = msg.imu_angles_rpy[0]
                theta_y = msg.imu_angles_rpy[1]
                theta_z = msg.imu_angles_rpy[2]
                psi_1 = 0
                psi_2 = 0
                psi_3 = 0
                dpsi_1 = 0
                dpsi_2 = 0
                dpsi_3 = 0
                
                # Store and printout data
                data = [i, t_now, Tx, Ty, Tz, u1, u2, u3, theta_x, theta_y, theta_z, psi_1, psi_2, psi_3, dpsi_1, dpsi_2, dpsi_3]
                dl.appendData(data)
                print(
                    f"Time: {t_now:.3f}s | Tx: {Tx:.2f}, Ty: {Ty:.2f}, Tz: {Tz:.2f} | "
                    f"u1: {u1:.2f}, u2: {u2:.2f}, u3: {u3:.2f} | "
                    f"Theta X: {theta_x:.2f}, Theta Y: {theta_y:.2f}, Theta Z: {theta_z:.2f} | "
                    f"Psi 1: {psi_1:.1f}, Psi 2: {psi_2:.1f}, Psi 3: {psi_3:.1f} | "
                    f"dPsi 1: {dpsi_1:.2f}, dPsi 2: {dpsi_2:.2f}, dPsi 3: {dpsi_3:.2f} | "
                )
            
            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping all commands...")
        # Emergency stop
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[0] = 0.0
        command.pwm[1] = 0.0
        command.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
    
    finally:
        # Save/log data
        print(f"Saving data as {filename}...")
        dl.writeOut()  # Write logged data to the file
        # Stop the listener thread
        listening = False
        print("Stopping LCM listener...")
        listener_thread.join(timeout=1)  # Wait up to 1 second for thread to finish

if __name__ == "__main__":
    main()
    