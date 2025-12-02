import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger3 import dataLogger
from ps4_controller_api import PS4InputHandler

# ===== helpers =====
from pid import PID

# Constants for the control loop
FREQ = 100  # Frequency of control loop [Hz]
DT = 1 / FREQ  # Time step for each iteration [sec]
PWM_MAX = 0.96
N_GEARBOX = 70
N_ENC = 64
R_W = 0.048
R_K = 0.121
alpha = 45*(np.pi/180)

# Steering/balance coordination
THETA_MAX_DEG = 3
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)
VEL_ACC_GAIN = 0.5   # a_des = VEL_ACC_GAIN * (v_des - v_meas)
VEL_FILT_ALPHA = 0.25  # EMA for measured ball velocities dPhi

# Steering safety: above this lean, ignore velocity commands & just balance
TH_STEER_PRIOR_DEG = 3.0
TH_STEER_PRIOR_RAD = np.deg2rad(TH_STEER_PRIOR_DEG)
VEL_CMD_EPS = 1e-3


# Set to True to enable balance mode (PID control on lean angles)
BALANCE_MODE = True

# Safety parameters
TH_ABORT_DEG = 25.0         # hard abort if tilt exceeds this (degrees)
PWM_USER_CAP = 0.96         # actuator saturation during balancing

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
    global listening, msg, BALANCE_MODE
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

    pid_x = PID(kp=8.6, ki=0, kd=0.02, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_y = PID(kp=8.6, ki=0, kd=0.02, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_x_steer = PID(kp=8.6, ki=0, kd=0.02, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)
    pid_y_steer = PID(kp=8.6, ki=0, kd=0.02, u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP, d_window=5)

    try:
        command = mbot_motor_pwm_t()
        print("Starting steering control loop...")
        time.sleep(0.5)
        # Logging header focused on PID tuning: errors, individual PID contributions, PID gains, PID outputs
                # Logging header compatible with DataLogger3_viewer
        header = [
            "i",
            "t_now",
            "Tx",
            "Ty",
            "Tz",
            "u1",
            "u2",
            "u3",
            "theta_x",
            "theta_y",
            "theta_z",
            "psi_1",
            "psi_2",
            "psi_3",
            "dpsi_1",
            "dpsi_2",
            "dpsi_3",
            # extra stuff you still want:
            "e_x",
            "e_y",
            "abort",
            "kp_x",
            "ki_x",
            "kd_x",
            "kp_y",
            "ki_y",
            "kd_y",
        ]
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
        # filtered ball velocity states
        vx_filt = 0.0
        vy_filt = 0.0
        
        # PS4 button state tracking for edge detection
        prev_tri = 0
        prev_cir = 0
        prev_x = 0
        
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
                
                # select gain to tune with dpad horizontal (left/right switches which gain)
                if dpad_h > prev_dpad_h:
                    cur_tune = (cur_tune + 1) % 3
                elif dpad_h < prev_dpad_h:
                    cur_tune = (cur_tune - 1) % 3
                
                #tune for balance mode
                if BALANCE_MODE:
                    # adjust selected gain with dpad vertical (up increases, down decreases)
                    if dpad_vert > prev_dpad_vert:
                        if cur_tune == 0:
                            pid_x.kp += step
                            pid_y.kp += step
                        elif cur_tune == 1:
                            pid_x.kd += step/10
                            pid_y.kd += step/10
                        else:
                            pid_x.ki += step/10
                            pid_y.ki += step/10
                    elif dpad_vert < prev_dpad_vert:
                        if cur_tune == 0:
                            pid_x.kp -= step
                            pid_y.kp -= step
                        elif cur_tune == 1:
                            pid_x.kd -= step/10
                            pid_y.kd -= step/10
                        else:
                            pid_x.ki -= step/10
                            pid_y.ki -= step/10
                #tune for steering
                else:
                    # adjust selected gain with dpad vertical (up increases, down decreases)
                    if dpad_vert > prev_dpad_vert:
                        if cur_tune == 0:
                            pid_x_steer.kp += step
                            pid_y_steer.kp += step
                        elif cur_tune == 1:
                            pid_x_steer.kd += step/10
                            pid_y_steer.kd += step/10
                        else:
                            pid_x_steer.ki += step/10
                            pid_y_steer.ki += step/10
                    elif dpad_vert < prev_dpad_vert:
                        if cur_tune == 0:
                            pid_x_steer.kp -= step
                            pid_y_steer.kp -= step
                        elif cur_tune == 1:
                            pid_x_steer.kd -= step/10
                            pid_y_steer.kd -= step/10
                        else:
                            pid_x_steer.ki -= step/10
                            pid_y_steer.ki -= step/10
                    
                    
                
                prev_dpad_h = dpad_h 
                prev_dpad_vert = dpad_vert
                # Safety buttons - Triangle kill, Circle IMU reset, and X to toggle balance/manual
                tri = bt_signals.get("but_tri", 0)
                cir = bt_signals.get("but_cir", 0)
                sqr = bt_signals.get("but_sqr", 0)
                but_x = bt_signals.get("but_x", 0)
                
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
                    pid_x_steer.reset()
                    pid_y_steer.reset()
                    print(f"IMU reset (Circle): theta_x_0={theta_x_0:.4f}, theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}")
                
                # X button toggles between balance and manual modes on press-edge
                if but_x and not prev_x:
                    BALANCE_MODE = not BALANCE_MODE
                    mode_str = "BALANCE" if BALANCE_MODE else "MANUAL"
                    print(f"Mode toggled (X): {mode_str}")

                prev_tri = tri
                prev_cir = cir
                prev_x = but_x

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
                # avoid div by zero
                enc_dt = max(enc_dt, 1e-6)
                dpsi_1 = calc_enc2rad(enc_dtick_1)/(enc_dt*1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2)/(enc_dt*1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3)/(enc_dt*1e-6)
                phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1,psi_2,psi_3)
                dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1,dpsi_2,dpsi_3)

                # filter measured ball velocities for steering 
                #vx_filt = (1.0 - VEL_FILT_ALPHA) * vx_filt + VEL_FILT_ALPHA * dphi_x
                #vy_filt = (1.0 - VEL_FILT_ALPHA) * vy_filt + VEL_FILT_ALPHA * dphi_y

                # Safety check - abort if too lean
                abort = too_lean(theta_x, theta_y, TH_ABORT_DEG)
                    
                now_t = time.time()
                dt = max(1e-6, now_t - prev_t)
                prev_t = now_t

                # ====================================================================
                # STEP 2A: JOYSTICK -> DESIRED VELOCITIES
                # ====================================================================
                DEADZONE  = 0.08

                js_fwd   = -js_R_y if abs(js_R_y) > DEADZONE else 0.0   # forward/back
                js_strafe = js_R_x if abs(js_R_x) > DEADZONE else 0.0   # left/right
                js_fwd = np.sign(js_fwd) * (js_fwd**2)
                js_strafe = np.sign(js_strafe) * (js_strafe**2)

                #vx_desired = VEL_SCALE * js_fwd      # forward velocity command (+x)
                #vy_desired = VEL_SCALE * js_strafe   # lateral velocity command (+y)

                # Measured velocities (filtered) from ball kinematics
                #vx_meas = vx_filt
                #vy_meas = vy_filt

                # ====================================================================
                # STEP 2B: DESIRED LEAN ANGLES (STEERING VIA LEAN ONLY)
                # ====================================================================
                # Default: pure balance (upright)
                theta_d_x = 0.0  # desired roll (for vy)
                theta_d_y = 0.0  # desired pitch (for vx)

                if not BALANCE_MODE:
                    # Command lean directly from joystick (no velocity loop)
                    if abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS:
                        theta_d_y = -THETA_MAX_RAD * js_fwd      # pitch  (vx)
                        theta_d_x =  THETA_MAX_RAD * js_strafe    # roll   (vy)
                    else:
                        theta_d_x = 0.0
                        theta_d_y = 0.0

                # Lean errors for logging (desired minus actual)
                e_x = theta_d_x - theta_x   # roll error
                e_y = theta_d_y - theta_y   # pitch error

                # Balance errors (upright target)
                e_bal_x = -theta_x
                e_bal_y = -theta_y

                # Steering errors (track commanded lean)
                e_str_x = e_x
                e_str_y = e_y

                Tx_balance = Ty_balance = 0.0
                Tx_steer   = Ty_steer   = 0.0
                Tx = Ty = Tz = 0.0

                if not abort:
                    # Always compute balance torques first
                    Tx_balance = pid_y.update(e_bal_y, dt)   # pitch -> Tx
                    Ty_balance = pid_x.update(e_bal_x, dt)   # roll  -> Ty

                    if BALANCE_MODE:
                        # Pure balance mode
                        pid_x_steer.reset()
                        pid_y_steer.reset()
                        Tx = func_clip(Tx_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Ty = func_clip(Ty_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Tz = 0.0
                    else:
                        # Steering via lean angle: add steering torques when safe and commanded
                        if (
                            (abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS)
                            and (abs(theta_x) < TH_STEER_PRIOR_RAD and abs(theta_y) < TH_STEER_PRIOR_RAD)
                        ):
                            Tx_steer = pid_y_steer.update(e_str_y, dt)
                            Ty_steer = pid_x_steer.update(e_str_x, dt)
                        else:
                            pid_x_steer.reset()
                            pid_y_steer.reset()
                            Tx_steer = 0.0
                            Ty_steer = 0.0

                        Tx = func_clip(Tx_balance + Tx_steer, -PWM_USER_CAP, PWM_USER_CAP)
                        Ty = func_clip(Ty_balance + Ty_steer, -PWM_USER_CAP, PWM_USER_CAP)
                        Tz = 0.0
                else:
                    # Emergency stop if too lean
                    Tx = Ty = Tz = 0.0
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()
                    print(
                        f"ABORT: Lean angle too high! "
                        f"theta_x={np.rad2deg(theta_x):.1f}°, theta_y={np.rad2deg(theta_y):.1f}°"
                    )

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


                # ======= Data Logging =======
                kp_x_term = pid_x.kp * e_x
                ki_x_term = pid_x.ki * pid_x._i
                kd_x_term = pid_x.kd * pid_x._d_filtered
                u_pid_x = Ty

                kp_y_term = pid_y.kp * e_y
                ki_y_term = pid_y.ki * pid_y._i
                kd_y_term = pid_y.kd * pid_y._d_filtered
                u_pid_y = Tx

                # Assemble data row matching the header order declared above
                data = [
                    i,                    # index
                    t_now,
                    float(Tx),
                    float(Ty),
                    float(Tz),
                    float(u1),
                    float(u2),
                    float(u3),
                    float(np.rad2deg(theta_x)),
                    float(np.rad2deg(theta_y)),
                    float(np.rad2deg(theta_z)),
                    float(psi_1),
                    float(psi_2),
                    float(psi_3),
                    float(dpsi_1),
                    float(dpsi_2),
                    float(dpsi_3),
                    # extra PID-related fields
                    float(np.rad2deg(e_x)),
                    float(np.rad2deg(e_y)),
                    float(abort),
                    float(pid_x.kp),
                    float(pid_x.ki),
                    float(pid_x.kd),
                    float(pid_y.kp),
                    float(pid_y.ki),
                    float(pid_y.kd),
                ]
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
                        f"[STEERING] t={t_now:6.3f}s | "
                        f"θx={np.rad2deg(theta_x):+6.2f}° (d={np.rad2deg(theta_d_x):+6.2f}°) | "
                        f"θy={np.rad2deg(theta_y):+6.2f}° (d={np.rad2deg(theta_d_y):+6.2f}°) | "
                        f"fwd_cmd={js_fwd:+.2f}, strf_cmd={js_strafe:+.2f} | "
                        f"Tx={Tx:+6.3f}, Ty={Ty:+6.3f} | "
                        f"Kx(Kp,Ki,Kd)=({pid_x.kp:.3f},{pid_x.ki:.4f},{pid_x.kd:.4f}) | "
                        f"Ky(Kp,Ki,Kd)=({pid_y.kp:.3f},{pid_y.ki:.4f},{pid_y.kd:.4f})"
                    )

                    # Indicate which gain is currently being tuned via D-pad
                    if cur_tune == 0:
                        print("    D-Pad tuning: Kp (both axes)")
                    elif cur_tune == 1:
                        print("    D-Pad tuning: Kd (both axes)")
                    else:
                        print("    D-Pad tuning: Ki (both axes)")

                    

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
