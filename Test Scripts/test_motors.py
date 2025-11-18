"""
ROB 311 - Fall 2025
Author: Prof. Greg Formosa
University of Michigan

Script is meant to test MBot Control Board motor drivers, without additional overhead.
Will store data as a "test_motors_[#].txt" file that can be parsed in Matlab to show motor PWM & encoder data.
Run script, choose a test number to store data as (will overwrite if file already exists), and then
watch motors increase/decrease in full range of speed. 

WARNING: be sure motors are in a safe orientation to run before starting!

Can edit PWM_MAX and PWM_PERIOD constants if desired, but PWM_MAX should be [0,1] and 
PWM_PERIOD > 1.5 seconds.

Press Ctrl+C to stop script at any time.

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
PWM_MAX = 1  # Max motor signal for full accel/decel of motor test
PWM_PERIOD = 3  # Time period for full accel/decel of motor test

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
    filename = f"test_motors_{trial_num}.txt"
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
        time.sleep(DT)
        data = ["i t_now u1 u2 u3 enc_1 enc_2 enc_3"]
        dl.appendData(data)
        i = 0  # Iteration counter
        t_start = time.time()
        t_now = 0
        enc_1_start = msg.enc_ticks[0]
        enc_2_start = msg.enc_ticks[1]
        enc_3_start = msg.enc_ticks[2]
        u1 = 0
        u2 = 0
        u3 = 0

        while (t_now < 2*PWM_PERIOD):
            time.sleep(DT)
            t_now = time.time() - t_start  # Elapsed time
            i += 1

            try:
                # Pull sensor data
                enc_1 = msg.enc_ticks[0] - enc_1_start
                enc_2 = msg.enc_ticks[1] - enc_2_start
                enc_3 = msg.enc_ticks[2] - enc_3_start
                
                # Calculate motor signals for test (ramp up cosine, ramp down cosine)
                u = -0.5*PWM_MAX*np.cos(2*np.pi*t_now/PWM_PERIOD) + 0.5*PWM_MAX
                if (t_now > PWM_PERIOD):
                    u = -u
                u1 = u
                u2 = u
                u3 = u

                # Send individual motor commands
                cmd_utime = int(time.time() * 1e6)
                command.utime = cmd_utime
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

                # Store and printout data
                data = [i, t_now, u1, u2, u3, enc_1, enc_2, enc_3]
                dl.appendData(data)
                print(
                    f"Time: {t_now:.3f}s | "
                    f"u1: {u1:.2f}, u2: {u2:.2f}, u3: {u3:.2f} | "
                    f"Enc 1: {enc_1}, Enc 2: {enc_2}, Enc 3: {enc_3} | "
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
    