"""
General framework for ball-bot control for students to update as desired.
You may wish to make multiple versions of this file to run your ball-bot in different modes!

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
from ps4_controller_api import PS4InputHandler

# Constants for the control loop
FREQ = 200  # Frequency of control loop [Hz]
DT = 1 / FREQ  # Time step for each iteration [sec]
PWM_MAX = 1  # Max motor signal for full accel/decel of motor test (keep between 0 and 1)
N_GEARBOX = 70 # Motor gearbox ratio
N_ENC = 64 # Ticks per revolution of encoder
R_W = 0.048 # Radii of omni-wheels [m]
R_K = 0.121 # Radius of basketball [m]
alpha = 45*(np.pi/180) # Angle of omni-wheels from vertical [rad]

# Global flags to control the listening thread & msg data
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0
last_seen = {"MBOT_BALBOT_FEEDBACK": 0}

def feedback_handler(channel, data):
    """Callback function to handle received mbot_balbot_feedback_t messages"""
    global msg
    global last_seen
    global last_time
    last_time = time.time()
    last_seen[channel] = time.time()
    msg = mbot_balbot_feedback_t.decode(data)

def lcm_listener(lc):
    """Function to continuously listen for LCM messages in a separate thread"""
    global listening
    while listening:
        try:
            lc.handle_timeout(100)  # 100ms timeout
            if time.time() - last_time > 2.0:
                print("LCM Publisher seems inactive...")
            elif time.time() - last_seen["MBOT_BALBOT_FEEDBACK"] > 2.0:
                print("LCM MBOT_BALBOT_FEEDBACK node seems inactive...")
        except Exception as e:
            print(f"LCM listening error: {e}")
            break


def calc_enc2rad(ticks):
    # TODO [LAB-08]: Calculate the angle (radians) of the wheel given the motor encoder ticks
    rad = (2*np.pi*ticks)/(N_ENC*N_GEARBOX)
    return rad

def calc_torque_conv(Tx,Ty,Tz):
    # Calculate desired motor torques T1,T2,T3 from given Tx,Ty,Tz
    u1 = (1/3)*(Tz- ((2*Ty)/np.cos(alpha)))  # motor 1
    u2 = (1/3)*(Tz + (1/np.cos(alpha))*(-np.sqrt(3)*Tx + Ty))  # motor 2
    u3 = (1/3)*(Tz + (1/np.cos(alpha))*(np.sqrt(3)*Tx + Ty)) # motor 3
    return u1, u2, u3

def calc_kinematic_conv(psi1,psi2,psi3):
    # TODO [LAB-08]: Calculate ball angular position from encoder odometry
    phi_x = (np.sqrt(2/3))*(R_W/R_K)*(psi2 - psi3)
    phi_y = (np.sqrt(2)/3)*(R_W/R_K)*(-2*psi1 + psi2 + psi3)
    phi_z = (np.sqrt(2)/3)*(R_W/R_K)*(psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z

def func_clip(x,lim_lo,lim_hi):
    # A function to clip values that exceed a threshold [lim_lo,lim_hi]
    if x > lim_hi:
        x = lim_hi
    elif x < lim_lo:
        x = lim_lo
    return x


def main():
    # === Data Logging Initialization ===
    # Prompt user for trial number and create a data logger
    trial_num = int(input("Test Number? "))
    filename = f"ballbot_encoder-imu_{trial_num}.txt"
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

    # === Controller Initialization ===
    # Create an instance of the PS4 controller handler
    controller = PS4InputHandler(interface="/dev/input/js0",connecting_using_ds4drv=False)
    # Start a separate thread to listen for controller inputs
    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.daemon = True  # Ensures the thread stops with the main program
    controller_thread.start()
    print("PS4 Controller is active...")

    try:
        command = mbot_motor_pwm_t()
        # === Main Control Loop ===
        print("Starting steering control loop...")
        time.sleep(0.5)
        # Store variable names as header to data logged, for easier parsing in Matlab
        # TODO [IF DESIRED]: Update data header variables names to match actual data logged (at end of loop)
        data = ["t_now Tz phi_x phi_y phi_z dphi_x dphi_y dphi_z enc_pos_1 enc_pos_2 enc_pos_3 enc_dtick_1 enc_dtick_2 enc_dtick_3 theta_x theta_y theta_z"]
        dl.appendData(data)
        i = 0  # Iteration counter
        t_start = time.time()
        t_now = 0
        enc_pos_1_start = msg.enc_ticks[0]
        enc_pos_2_start = msg.enc_ticks[1]
        enc_pos_3_start = msg.enc_ticks[2]
        u1 = 0
        u2 = 0
        u3 = 0
        theta_x_0 = 0
        theta_y_0 = 0
        theta_z_0 = 0

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start  # Elapsed time
            i += 1

            try:
                # retreive dictionary of button press signals from handler
                bt_signals = controller.get_signals()
                # parse out individual buttons you want data from
                js_R_x = bt_signals["js_R_x"]   # steering bot (XY) with js_R
                js_R_y = bt_signals["js_R_y"]
                js_L_x = bt_signals["js_L_x"]   # steering bot (XY) with js_L
                js_L_y = bt_signals["js_L_y"]
                trigger_L2 = bt_signals["trigger_L2"]   # spinning bot (Z) with L2/R2 triggers
                trigger_R2 = bt_signals["trigger_R2"]

                # Pull sensor data
                theta_x = msg.imu_angles_rpy[0] - theta_x_0
                theta_y = msg.imu_angles_rpy[1] - theta_y_0
                theta_z = msg.imu_angles_rpy[2] - theta_z_0
                enc_pos_1 = msg.enc_ticks[0] - enc_pos_1_start
                enc_pos_2 = msg.enc_ticks[1] - enc_pos_2_start
                enc_pos_3 = msg.enc_ticks[2] - enc_pos_3_start
                enc_dtick_1 = msg.enc_delta_ticks[0]
                enc_dtick_2 = msg.enc_delta_ticks[1]
                enc_dtick_3 = msg.enc_delta_ticks[2]
                enc_dt = msg.enc_delta_time

                # Calculate motor angles from encoder ticks
                # TODO [LAB-08]: Call method to calculate motor angles & speeds from measured encoder values
                psi_1 = calc_enc2rad(enc_pos_1)
                psi_2 = calc_enc2rad(enc_pos_2)
                psi_3 = calc_enc2rad(enc_pos_3)
                dpsi_1 = calc_enc2rad(enc_dtick_1)/(enc_dt*1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2)/(enc_dt*1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3)/(enc_dt*1e-6)

                # Calculate ball's roll and translation through kinematic conversions of wheel data
                # TODO [LAB-08]: Call method to calculate kinematic conversion of encoder-angles to ball-translations
                phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1,psi_2,psi_3)
                dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1,dpsi_2,dpsi_3)

                # Set x-y-z bot commands
                # TODO [LAB-07 & LAB-08]: Choose how Tx,Ty,Tz are set

                #floor mapping 
                # Tx = -1 => +y direction
                # Tx = 1 => -y direction
                # Ty = 1 => -x direction
                # Ty = -1 => +x direction

                if t_now < 3:
                    # 0–3 seconds → slow rotation
                    Tx, Ty, Tz = 0, 0, 0.75
                elif t_now < 6:
                    # 3–6 seconds → medium rotation
                    Tx, Ty, Tz = 0, 0, 1.5
                elif t_now < 9:
                    # 6–9 seconds → fast rotation
                    Tx, Ty, Tz = 0, 0, 2.25
                elif t_now < 12:
                    Tx, Ty, Tz = 0, 0, 3
                else:
                    print("Test complete (12s reached). Stopping motors...")
                    command.utime = int(time.time() * 1e6)
                    command.pwm[:] = [0.0, 0.0, 0.0]  # remember to set the pwm commands to zero    
                    lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                    break
                
                # Calculate motor effort/commands from desired Tx,Ty,Tz motion
                # TODO [LAB-07]: Call method to calculate motor commands (u1,u2,u3) from axis torques (Tx,Ty,Tz)
                u1,u2,u3 = calc_torque_conv(Tx,Ty,Tz)
                
                # Send individual motor commands
                u1 = func_clip(u1,-PWM_MAX,PWM_MAX)
                u2 = func_clip(u2,-PWM_MAX,PWM_MAX)
                u3 = func_clip(u3,-PWM_MAX,PWM_MAX)
                cmd_utime = int(time.time() * 1e6)
                command.utime = cmd_utime
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                
                # Store data in data logger
                data = [t_now, Tz, phi_x, phi_y, phi_z, dphi_x, dphi_y, dphi_z, enc_pos_1, enc_pos_2, enc_pos_3, enc_dtick_1, enc_dtick_2, enc_dtick_3, theta_x, theta_y, theta_z]
                # TODO [IF DESIRED]: Update variables to match data header names for logging
                # 
                # dl.appendData(data)
                # # Print out data in terminal
                # # TODO: [IF DESIRED]: Update for what info you want to see in terminal (note: this is only printed data, not logged!)
                # print(
                #     f"Time: {t_now:.3f}s | Tx: {Tx:.2f}, Ty: {Ty:.2f}, Tz: {Tz:.2f} | "
                #     f"u1: {u1:.2f}, u2: {u2:.2f}, u3: {u3:.2f} | "
                #     f"Theta X: {theta_x:.2f}, Theta Y: {theta_y:.2f}, Theta Z: {theta_z:.2f} | "
                #     f"Psi 1: {enc_pos_1:.1f}, Psi 2: {enc_pos_2:.1f}, Psi 3: {enc_pos_3:.1f} | "
                #     f"dPsi 1: {enc_dtick_1:.2f}, dPsi 2: {enc_dtick_2:.2f}, dPsi 3: {enc_dtick_3:.2f} | "
                # )

                # For Lab 8 - on the floor
                data = [t_now, Tx, Ty, Tz, u1, u2, u3, dpsi_1, dpsi_2, dpsi_3, phi_x, phi_y, phi_z, dphi_x, dphi_y, dphi_z]
                dl.appendData(data)
                # Print out data in terminal
                print(
                    f"Time: {t_now:.3f}s | Tx: {Tx:.2f}, Ty: {Ty:.2f}, Tz: {Tz:.2f} | "
                    f"u1: {u1:.2f}, u2: {u2:.2f}, u3: {u3:.2f} | "
                    f"dPsi 1: {dpsi_1:.2f}, dPsi 2: {dpsi_2:.2f}, dPsi 3: {dpsi_3:.2f} | "
                    f"Phi X: {phi_x:.2f}, Phi Y: {phi_y:.2f}, Phi Z: {phi_z:.2f} | "
                    f"dPhi X: {dphi_x:.2f}, dPhi Y: {dphi_y:.2f}, dPhi Z: {dphi_z:.2f} | "
                )
            
            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping motors...")
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
        # Stop Bluetooth thread
        controller_thread.join(timeout=1)  # Wait up to 1 second for thread to finish
        controller.on_options_press()
        # Stop motors
        print("Shutting down motors...\n")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[0] = 0.0
        command.pwm[1] = 0.0
        command.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

if __name__ == "__main__":
    main()