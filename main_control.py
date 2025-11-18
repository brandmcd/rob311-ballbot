import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger2 import dataLogger
from ps4_controller_api import PS4InputHandler

# ===== helpers =====
from pid import PID

# Constants for the control loop
FREQ = 200  # Frequency of control loop [Hz]
DT = 1 / FREQ  # Time step for each iteration [sec]
PWM_MAX = 1
N_GEARBOX = 70
N_ENC = 64
R_W = 0.048
R_K = 0.121
alpha = 45*(np.pi/180)

# Set to True to enable balance mode (PID control on lean angles)
BALANCE_MODE = True

# Safety parameters
TH_ABORT_DEG = 25.0         # hard abort if tilt exceeds this (degrees)
PWM_USER_CAP = 0.98         # actuator saturation during balancing

# Global flags to control the listening thread & msg data
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0
last_seen = {"MBOT_BALBOT_FEEDBACK": 0}

def feedback_handler(channel, data):
    global msg, last_seen, last_time
    last_time = time.time()
    last_seen[channel] = time.time()
    msg = mbot_balbot_feedback_t.decode(data)


def lcm_listener(lc):
    global listening
    while listening:
        try:
            lc.handle_timeout(100)
            if time.time() - last_time > 2.0:
                print("LCM Publisher seems inactive...")
            elif time.time() - last_seen["MBOT_BALBOT_FEEDBACK"] > 2.0:
                print("LCM MBOT_BALBOT_FEEDBACK node seems inactive...")
        except Exception as e:
            print(f"LCM listening error: {e}")
            break


def calc_enc2rad(ticks):
    return (2*np.pi*ticks)/(N_ENC*N_GEARBOX)


def calc_torque_conv(Tx,Ty,Tz):
    u1 = (1/3)*(Tz- ((2*Ty)/np.cos(alpha)))
    u2 = (1/3)*(Tz + (1/np.cos(alpha))*(-np.sqrt(3)*Tx + Ty))
    u3 = (1/3)*(Tz + (1/np.cos(alpha))*(np.sqrt(3)*Tx + Ty))
    return u1, u2, u3


def calc_kinematic_conv(psi1,psi2,psi3):
    phi_x = (np.sqrt(2/3))*(R_W/R_K)*(psi2 - psi3)
    phi_y = (np.sqrt(2)/3)*(R_W/R_K)*(-2*psi1 + psi2 + psi3)
    phi_z = (np.sqrt(2)/3)*(R_W/R_K)*(psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z


def func_clip(x,lim_lo,lim_hi):
    if x > lim_hi: x = lim_hi
    elif x < lim_lo: x = lim_lo
    return x


def too_lean(theta_x, theta_y, deg_limit):
    """Return True if the robot is leaned beyond deg_limit (in degrees)."""
    return (abs(theta_x) > np.deg2rad(deg_limit)) or (abs(theta_y) > np.deg2rad(deg_limit))


def main():
    # === Data Logging Initialization ===
    trial_num = int(input("Test Number? "))
    filename = f"main_control_{trial_num}.txt"
    dl = dataLogger(filename)

    # === LCM Messaging Initialization ===
    global listening, msg
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("Started continuous LCM listener...")

    # === Controller Initialization ===
    controller = PS4InputHandler(interface="/dev/input/js0",connecting_using_ds4drv=False)
    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.daemon = True
    controller_thread.start()
    print("PS4 Controller is active...")
    print("Controls: Triangle=KILL, Circle=Reset IMU, L/R sticks=manual control (when BALANCE_MODE=False)")
    #Kp_x: 9.300, Ki_x: 0.4300, Kd_x: 0.0500
    pid_x = PID(kp=7.7, ki=0, kd=0, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_y = PID(kp=7.7, ki=0, kd=0, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_x_steer = PID(kp=7.7, ki=0, kd=0, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_y_steer = PID(kp=7.7, ki=0, kd=0, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)

    try:
        command = mbot_motor_pwm_t()
        print("Starting steering control loop...")
        time.sleep(0.5)
        header = ["t_now","Tx","Ty","Tz","u1","u2","u3","dpsi_1","dpsi_2","dpsi_3","phi_x","phi_y","phi_z","dphi_x","dphi_y","dphi_z","theta_x","theta_y","theta_z","abort","kp","ki","kd"]
        dl.appendData(header)

        i = 0
        t_start = time.time()
        t_now = 0
        enc_pos_1_start = msg.enc_ticks[0]
        enc_pos_2_start = msg.enc_ticks[1]
        enc_pos_3_start = msg.enc_ticks[2]
        u1 = u2 = u3 = 0
        # zero angles on first sample per lab note
        theta_x_0 = msg.imu_angles_rpy[0]
        theta_y_0 = msg.imu_angles_rpy[1]
        theta_z_0 = msg.imu_angles_rpy[2]
        print(f"IMU offsets set: theta_x_0={theta_x_0:.4f}, theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}")
        prev_t = time.time()
        
        # PS4 button state tracking for edge detection
        prev_tri = 0
        prev_cir = 0
        
        # D-pad tuning variables
        step = 0.1
        cur_tune = 0  # 0 for Kp, 1 for Kd, 2 for Ki
        prev_dpad_h = 0
        prev_dpad_vert = 0

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start
            i += 1

            try:
                # gamepad
                bt_signals = controller.get_signals()
                js_R_x = bt_signals["js_R_x"]
                js_R_y = bt_signals["js_R_y"]
                js_L_x = bt_signals["js_L_x"]
                js_L_y = bt_signals["js_L_y"]
                trigger_L2 = bt_signals["trigger_L2"]
                trigger_R2 = bt_signals["trigger_R2"]
                
                # D-pad tuning
                dpad_h = bt_signals["dpad_horiz"]
                dpad_vert = bt_signals["dpad_vert"]
                
                # select gain to tune with dpad vertical
                if dpad_vert > prev_dpad_vert:
                    cur_tune = (cur_tune + 1) % 3
                        
                    
                # adjust gain with dpad horizontal
                if dpad_h > prev_dpad_h:
                    if cur_tune == 0:
                        pid_x.kp += step
                        pid_y.kp += step
                    elif cur_tune == 1:
                        pid_x.kd += step/10
                        pid_y.kd += step/10
                    else:
                        pid_x.ki += step/10
                        pid_y.ki += step/10
                elif dpad_h < prev_dpad_h:
                    if cur_tune == 0:
                        pid_x.kp -= step
                        pid_y.kp -= step
                    elif cur_tune == 1:
                        pid_x.kd -= step/10
                        pid_y.kd -= step/10
                    else:
                        pid_x.ki -= step/10
                        pid_y.ki -= step/10
                    
                
                prev_dpad_h = dpad_h 
                prev_dpad_vert = dpad_vert
                # Safety buttons - Triangle kill and Circle IMU reset
                tri = bt_signals.get("but_tri", 0)
                cir = bt_signals.get("but_cir", 0)
                
                # Triangle kill (immediate stop)
                if tri and not prev_tri:
                    print("PS4 KILL (Triangle) pressed - stopping motors and exiting.")
                    command = mbot_motor_pwm_t()
                    command.utime = int(time.time() * 1e6)
                    command.pwm[:] = [0.0, 0.0, 0.0]
                    lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                    break
                
                # Circle IMU reset
                if cir and not prev_cir:
                    theta_x_0 = msg.imu_angles_rpy[0]
                    theta_y_0 = msg.imu_angles_rpy[1]
                    theta_z_0 = msg.imu_angles_rpy[2]
                    pid_x.reset()
                    pid_y.reset()
                    print(f"IMU reset (Circle): theta_x_0={theta_x_0:.4f}, theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}")
                
                prev_tri = tri
                prev_cir = cir

                # sensors
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

                # wheel kinematics
                psi_1 = calc_enc2rad(enc_pos_1)
                psi_2 = calc_enc2rad(enc_pos_2)
                psi_3 = calc_enc2rad(enc_pos_3)
                dpsi_1 = calc_enc2rad(enc_dtick_1)/(enc_dt*1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2)/(enc_dt*1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3)/(enc_dt*1e-6)
                phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1,psi_2,psi_3)
                dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1,dpsi_2,dpsi_3)

                # Safety check - abort if too lean
                abort = too_lean(theta_x, theta_y, TH_ABORT_DEG)
                    
                now_t = time.time()
                dt = max(1e-6, now_t - prev_t)
                prev_t = now_t

                e_x = -theta_x  
                e_y = -theta_y  

                # Get desired velocity from user
                vx_desired = 0 #js_R_y   # Want to move forward
                vy_desired = 0 #js_R_x   # Want to strafe right
                #recalc to be world frame
                #temp = vx_desired * np.cos(theta_z) - vy_desired * np.sin(theta_z)
                #vy_desired = vx_desired * np.sin(theta_z) + vy_desired * np.cos(theta_z)
                #vx_desired = temp

                # Calculate velocity errors
                error_vx = vx_desired - dphi_x
                error_vy = vy_desired - dphi_y
                
                if BALANCE_MODE and not abort:
                    Tx = pid_y.update(e_y, dt)
                    Ty = pid_x.update(e_x, dt)
                    Tz = 0.0  # keep yaw idle while balancing
                # Manual Mode
                elif not abort:
                    Tx_steering = pid_x_steer.update(error_vx, dt)
                    Ty_steering = pid_y_steer.update(error_vy, dt)
                    Tx_balance = pid_y.update(e_y, dt)
                    Ty_balance = pid_x.update(e_x, dt)
                    Tx = Tx_balance + (0.5 * Tx_steering)
                    Ty = Ty_balance + (0.5 * Ty_steering)
                    Tz = 0.0
                elif abort:
                    # Emergency stop if too lean
                    Tx = Ty = Tz = 0.0
                    pid_x.reset()
                    pid_y.reset()
                    print(f"ABORT: Lean angle too high! theta_x={np.rad2deg(theta_x):.1f}°, theta_y={np.rad2deg(theta_y):.1f}°")


                # motor commands
                u1,u2,u3 = calc_torque_conv(Tx,Ty,Tz)
                u1 = func_clip(u1,-PWM_MAX,PWM_MAX)
                u2 = func_clip(u2,-PWM_MAX,PWM_MAX)
                u3 = func_clip(u3,-PWM_MAX,PWM_MAX)
                cmd_utime = int(time.time() * 1e6)
                command.utime = cmd_utime
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

                # logging
                data = [t_now, Tx, Ty, Tz, u1, u2, u3, dpsi_1, dpsi_2, dpsi_3, phi_x, phi_y, phi_z, dphi_x, dphi_y, dphi_z, 
                       np.rad2deg(theta_x), np.rad2deg(theta_y), np.rad2deg(theta_z), abort, pid_x.kp, pid_x.ki, pid_x.kd]
                data = [float(v) for v in data]
                dl.appendData(data)
                
                if BALANCE_MODE:
                    abort_str = " [ABORT]" if abort else ""
                    print(
                        f"Time: {t_now:.3f}s{abort_str} | θx: {np.rad2deg(theta_x):+6.2f}°, θy: {np.rad2deg(theta_y):+6.2f}° | "
                        f"Kp_x: {pid_x.kp:.3f}, Ki_x: {pid_x.ki:.4f}, Kd_x: {pid_x.kd:.4f} | "
                        f"Kp_y: {pid_y.kp:.3f}, Ki_y: {pid_y.ki:.4f}, Kd_y: {pid_y.kd:.4f} | "
                        f"Tx: {Tx:+6.3f}, Ty: {Ty:+6.3f} | u1: {u1:+6.3f}, u2: {u2:+6.3f}, u3: {u3:+6.3f}"
                    )
                    if cur_tune == 0:
                        print("Tuning Kp")
                    elif cur_tune == 1:
                        print("Tuning Kd")
                    else:
                        print("Tuning Ki")
                else:
                    print(
                        f"Time: {t_now:.3f}s | Manual Mode | "
                        f"theta_x: {np.rad2deg(theta_x):.2f}°, theta_y: {np.rad2deg(theta_y):.2f}°, theta_z: {np.rad2deg(theta_z):.2f}° | "
                        f"vx_des: {vx_desired:.2f}, vy_des: {vy_desired:.2f} | "
                        f"Tx_steer: {Tx_steering:.2f}, Ty_steer: {Ty_steering:.2f} | "
                        f"Tx_balance: {Tx_balance:.2f}, Ty_balance: {Ty_balance:.2f} | "
                        f"u1: {u1:.2f}, u2: {u2:.2f}, u3: {u3:.2f} | "
                        f"dPhi X: {dphi_x:.2f}, dPhi Y: {dphi_y:.2f}, dPhi Z: {dphi_z:.2f} | "
                    )

            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping motors...")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[0] = 0.0
        command.pwm[1] = 0.0
        command.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

    finally:
        print(f"Saving data as {filename}...")
        dl.writeOut()
        listening = False
        print("Stopping LCM listener...")
        listener_thread.join(timeout=1)
        controller_thread.join(timeout=1)
        controller.on_options_press()
        print("Shutting down motors...")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[0] = 0.0
        command.pwm[1] = 0.0
        command.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

if __name__ == "__main__":
    main()
