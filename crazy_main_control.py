"""
Ballbot Main Control Script
============================
Balances a ballbot using PID control with optional drift correction.

Controls:
  Triangle: Emergency stop
  Circle: Reset IMU & encoders
  X: Toggle drift correction ON/OFF
  Square: Toggle Balance/Steering mode
  D-Pad: Tune PID gains (↑/↓ adjust, ←/→ switch parameter)
  Left Stick (horizontal): Tz turning/spinning (both modes)
  Right Stick: Steering (when in steering mode)
"""

import time
import lcm
import threading
import numpy as np
from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t
from DataLogger3 import dataLogger
from ps4_controller_api import PS4InputHandler
from pid import PID

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Control loop timing
FREQ = 100
DT = 1 / FREQ
PWM_MAX = 0.96
PWM_USER_CAP = 0.96

# Robot geometry
N_GEARBOX = 70
N_ENC = 64
R_W = 0.048  # Wheel radius [m]
R_K = 0.121  # Ball radius [m]
ALPHA = 45 * (np.pi/180)

# Balance controller (inner loop - angle stabilization)
BALANCE_KP = 8.5
BALANCE_KI = 20
BALANCE_KD = 0.2
BALANCE_INTEG_MIN = -0.15
BALANCE_INTEG_MAX = 100

# Drift correction (outer loop - position stabilization)
# Penalizes movement to keep robot stationary while balancing
DRIFT_KP = 0.08          # Position feedback - penalize displacement from center
DRIFT_KD = 0.015         # Velocity damping - penalize movement speed
DRIFT_MAX_ANGLE = 0.02   # Max correction angle: ~1.15°
DRIFT_POS_DEADZONE = 0.01  # Position deadzone in radians (~0.57°) - ignore small movements
DRIFT_VEL_DEADZONE = 0.02  # Velocity deadzone in rad/s - ignore small velocities

# Steering controller - using high kI logic like balance controller
STEER_KP = 3.0           # Much lower - was causing oscillations at 10.0
STEER_KI = 7.0           # High integral like balance controller for steady state
STEER_KD = 0.05          # More damping to prevent oscillations
STEER_INTEG_MIN = -0.15
STEER_INTEG_MAX = 100
THETA_MAX_DEG = 2.0      # Larger max lean for steering (was 0.5° - too small!)
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)
TH_STEER_PRIOR_RAD = np.deg2rad(3.0)  # Balance priority threshold
VEL_CMD_EPS = 1e-3
TZ_MAX = 0.3             # Maximum Tz (spinning) torque PWM value

# Safety
TH_ABORT_DEG = 25.0  # Emergency stop threshold

# Global state
listening = False
msg = mbot_balbot_feedback_t()
last_time = 0
last_seen = {"MBOT_BALBOT_FEEDBACK": 0}
BALANCE_MODE = True
DRIFT_CORRECTION_ENABLED = False  # Start disabled - enable after placing on ball

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def feedback_handler(channel, data):
    """LCM callback for sensor feedback."""
    global msg, last_seen, last_time
    last_time = time.time()
    last_seen[channel] = time.time()
    msg = mbot_balbot_feedback_t.decode(data)

def lcm_listener(lc):
    """Background thread for LCM message handling."""
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
    """Convert encoder ticks to radians."""
    return (2 * np.pi * ticks) / (N_ENC * N_GEARBOX)

def calc_torque_conv(Tx, Ty, Tz):
    """Convert body torques to wheel torques."""
    u1 = (1/3) * (Tz - ((2*Ty)/np.cos(ALPHA)))
    u2 = (1/3) * (Tz + (1/np.cos(ALPHA)) * (-np.sqrt(3)*Tx + Ty))
    u3 = (1/3) * (Tz + (1/np.cos(ALPHA)) * (np.sqrt(3)*Tx + Ty))
    return u1, u2, u3

def calc_kinematic_conv(psi1, psi2, psi3):
    """Convert wheel angles to ball position/orientation."""
    phi_x = (np.sqrt(2/3)) * (R_W/R_K) * (psi2 - psi3)
    phi_y = (np.sqrt(2)/3) * (R_W/R_K) * (-2*psi1 + psi2 + psi3)
    phi_z = (np.sqrt(2)/3) * (R_W/R_K) * (psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z

def func_clip(x, lim_lo, lim_hi):
    """Clip value to range."""
    return max(lim_lo, min(lim_hi, x))

def too_lean(theta_x, theta_y, deg_limit):
    """Check if robot is leaning beyond safe threshold."""
    return (abs(theta_x) > np.deg2rad(deg_limit)) or (abs(theta_y) > np.deg2rad(deg_limit))

# ============================================================================
# MAIN CONTROL LOOP
# ============================================================================

def main():
    global listening, msg, BALANCE_MODE, DRIFT_CORRECTION_ENABLED

    # === Setup ===
    trial_num = int(input("Test Number? "))
    filename = f"main_control_{trial_num}.txt"
    dl = dataLogger(filename)

    # LCM communication
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    subscription = lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("Started LCM listener...")

    # PS4 controller
    controller = PS4InputHandler(interface="/dev/input/js0", connecting_using_ds4drv=False)
    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.daemon = True
    controller_thread.start()
    print("PS4 Controller active...")
    print("\nControls:")
    print("  Triangle = Emergency Stop")
    print("  Circle = Reset IMU & PIDs & Zero Encoders")
    print("  X = Toggle Drift Correction ON/OFF")
    print("  Square = Toggle Balance/Steering Mode")
    print("  Left Stick (horizontal) = Tz turning/spinning (both modes)")
    print("  Right Stick = Steering (when in steering mode)")
    print("  D-Pad = Tune PID gains")

    # PID controllers
    pid_x = PID(
        kp=BALANCE_KP, ki=BALANCE_KI, kd=BALANCE_KD,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=BALANCE_INTEG_MIN, integ_max=BALANCE_INTEG_MAX,
        d_window=5, enable_antiwindup=True
    )
    pid_y = PID(
        kp=BALANCE_KP, ki=BALANCE_KI, kd=BALANCE_KD,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=BALANCE_INTEG_MIN, integ_max=BALANCE_INTEG_MAX,
        d_window=5, enable_antiwindup=True
    )
    pid_x_steer = PID(
        kp=STEER_KP, ki=STEER_KI, kd=STEER_KD,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=STEER_INTEG_MIN, integ_max=STEER_INTEG_MAX,
        d_window=5, enable_antiwindup=True
    )
    pid_y_steer = PID(
        kp=STEER_KP, ki=STEER_KI, kd=STEER_KD,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=STEER_INTEG_MIN, integ_max=STEER_INTEG_MAX,
        d_window=5, enable_antiwindup=True
    )

    try:
        command = mbot_motor_pwm_t()
        print("\nStarting control loop...")
        time.sleep(0.5)

        # Data logging header
        header = [
            "i", "t_now", "Tx", "Ty", "Tz", "u1", "u2", "u3",
            "theta_x", "theta_y", "theta_z", "theta_d_x", "theta_d_y",
            "phi_x", "phi_y", "dphi_x", "dphi_y",
            "psi_1", "psi_2", "psi_3", "dpsi_1", "dpsi_2", "dpsi_3",
            "e_x", "e_y", "abort",
            "kp_x", "ki_x", "kd_x", "kp_y", "ki_y", "kd_y",
            "p_x", "i_x", "d_x", "p_y", "i_y", "d_y",
        ]
        dl.appendData(header)

        # Initialize state
        i = 0
        t_start = time.time()
        prev_t = time.time()
        time.sleep(0.1)  # Wait for first message

        # Zero sensors
        enc_pos_1_start = msg.enc_ticks[0]
        enc_pos_2_start = msg.enc_ticks[1]
        enc_pos_3_start = msg.enc_ticks[2]
        theta_x_0 = msg.imu_angles_rpy[0]
        theta_y_0 = msg.imu_angles_rpy[1]
        theta_z_0 = msg.imu_angles_rpy[2]
        print(f"IMU offsets: theta_x_0={theta_x_0:.4f}, theta_y_0={theta_y_0:.4f}, theta_z_0={theta_z_0:.4f}")

        # Button states
        prev_tri = prev_cir = prev_x = prev_sq = 0
        prev_dpad_h = prev_dpad_vert = 0

        # Tuning
        tune_step = 0.1
        cur_tune = 0  # 0=Kp, 1=Ki, 2=Kd
        tune_names = ["Kp", "Ki", "Kd"]

        print("\n=== BALANCE MODE - Drift correction DISABLED ===")
        print("    Place robot on ball, press Circle to zero, then press X to enable drift correction\n")

        # ========================================================================
        # MAIN CONTROL LOOP
        # ========================================================================
        while True:
            time.sleep(DT)
            t_now = time.time() - t_start
            i += 1

            try:
                # ================================================================
                # READ GAMEPAD INPUTS
                # ================================================================
                bt_signals = controller.get_signals()
                js_R_x = bt_signals["js_R_x"]
                js_R_y = bt_signals["js_R_y"]
                js_L_x = bt_signals["js_L_x"]
                js_L_y = bt_signals["js_L_y"]
                dpad_h = bt_signals["dpad_horiz"]
                dpad_vert = bt_signals["dpad_vert"]
                tri = bt_signals.get("but_tri", 0)
                cir = bt_signals.get("but_cir", 0)
                but_x = bt_signals.get("but_x", 0)
                but_sq = bt_signals.get("but_sq", 0)

                # D-pad: Select which parameter to tune
                if dpad_h > prev_dpad_h:
                    cur_tune = (cur_tune + 1) % 3
                elif dpad_h < prev_dpad_h:
                    cur_tune = (cur_tune - 1) % 3

                # D-pad: Adjust selected parameter
                target_pids = [pid_x, pid_y] if BALANCE_MODE else [pid_x_steer, pid_y_steer]
                if dpad_vert > prev_dpad_vert:
                    for pid in target_pids:
                        if cur_tune == 0:
                            pid.kp += tune_step
                        elif cur_tune == 1:
                            pid.ki += tune_step/10
                        else:
                            pid.kd += tune_step/10
                elif dpad_vert < prev_dpad_vert:
                    for pid in target_pids:
                        if cur_tune == 0:
                            pid.kp -= tune_step
                        elif cur_tune == 1:
                            pid.ki -= tune_step/10
                        else:
                            pid.kd -= tune_step/10

                prev_dpad_h = dpad_h
                prev_dpad_vert = dpad_vert

                # Triangle: Emergency stop
                if tri and not prev_tri:
                    print("Emergency stop (Triangle) - exiting")
                    command.utime = int(time.time() * 1e6)
                    command.pwm[:] = [0.0, 0.0, 0.0]
                    lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())
                    break

                # Circle: Reset IMU and encoders
                if cir and not prev_cir:
                    theta_x_0 = msg.imu_angles_rpy[0]
                    theta_y_0 = msg.imu_angles_rpy[1]
                    theta_z_0 = msg.imu_angles_rpy[2]
                    enc_pos_1_start = msg.enc_ticks[0]
                    enc_pos_2_start = msg.enc_ticks[1]
                    enc_pos_3_start = msg.enc_ticks[2]
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()
                    print(f"Reset (Circle): θ0=[{theta_x_0:.3f}, {theta_y_0:.3f}, {theta_z_0:.3f}]")

                # X: Toggle drift correction
                if but_x and not prev_x:
                    DRIFT_CORRECTION_ENABLED = not DRIFT_CORRECTION_ENABLED
                    status = "ENABLED" if DRIFT_CORRECTION_ENABLED else "DISABLED"
                    print(f"Drift correction (X): {status}")

                # Square: Toggle balance/steering mode
                if but_sq and not prev_sq:
                    BALANCE_MODE = not BALANCE_MODE
                    mode = "BALANCE" if BALANCE_MODE else "STEERING"
                    print(f"Mode (Square): {mode}")

                prev_tri, prev_cir, prev_x, prev_sq = tri, cir, but_x, but_sq

                # ================================================================
                # READ SENSORS
                # ================================================================
                # IMU angles (zeroed)
                theta_x = msg.imu_angles_rpy[0] - theta_x_0
                theta_y = msg.imu_angles_rpy[1] - theta_y_0
                theta_z = msg.imu_angles_rpy[2] - theta_z_0

                # Encoder positions (zeroed)
                enc_pos_1 = msg.enc_ticks[0] - enc_pos_1_start
                enc_pos_2 = msg.enc_ticks[1] - enc_pos_2_start
                enc_pos_3 = msg.enc_ticks[2] - enc_pos_3_start
                enc_dtick_1 = msg.enc_delta_ticks[0]
                enc_dtick_2 = msg.enc_delta_ticks[1]
                enc_dtick_3 = msg.enc_delta_ticks[2]
                enc_dt = max(msg.enc_delta_time, 1e-6)

                # Wheel angles and velocities
                psi_1 = calc_enc2rad(enc_pos_1)
                psi_2 = calc_enc2rad(enc_pos_2)
                psi_3 = calc_enc2rad(enc_pos_3)
                dpsi_1 = calc_enc2rad(enc_dtick_1) / (enc_dt * 1e-6)
                dpsi_2 = calc_enc2rad(enc_dtick_2) / (enc_dt * 1e-6)
                dpsi_3 = calc_enc2rad(enc_dtick_3) / (enc_dt * 1e-6)

                # Ball position and velocity
                phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1, psi_2, psi_3)
                dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1, dpsi_2, dpsi_3)

                # Safety check
                abort = too_lean(theta_x, theta_y, TH_ABORT_DEG)

                # Time step
                now_t = time.time()
                dt = max(1e-6, now_t - prev_t)
                prev_t = now_t

                # ================================================================
                # COMPUTE CONTROL
                # ================================================================
                # Joystick processing
                DEADZONE = 0.08
                js_fwd = -js_R_y if abs(js_R_y) > DEADZONE else 0.0
                js_strafe = js_R_x if abs(js_R_x) > DEADZONE else 0.0
                js_spin = js_L_x if abs(js_L_x) > DEADZONE else 0.0
                js_fwd = np.sign(js_fwd) * (js_fwd**2)
                js_strafe = np.sign(js_strafe) * (js_strafe**2)
                js_spin = np.sign(js_spin) * (js_spin**2)

                # Compute desired tilt angles
                if BALANCE_MODE:
                    if DRIFT_CORRECTION_ENABLED:
                        # Drift correction with deadzone: penalize movement to stay stationary
                        # Apply position feedback only if outside deadzone
                        pos_x = -phi_x if abs(phi_x) > DRIFT_POS_DEADZONE else 0.0
                        pos_y = -phi_y if abs(phi_y) > DRIFT_POS_DEADZONE else 0.0

                        # Apply velocity damping only if outside deadzone
                        vel_x = -dphi_x if abs(dphi_x) > DRIFT_VEL_DEADZONE else 0.0
                        vel_y = -dphi_y if abs(dphi_y) > DRIFT_VEL_DEADZONE else 0.0

                        # Combine position and velocity feedback
                        drift_x = DRIFT_KP * pos_x + DRIFT_KD * vel_x
                        drift_y = DRIFT_KP * pos_y + DRIFT_KD * vel_y

                        theta_d_x = func_clip(drift_x, -DRIFT_MAX_ANGLE, DRIFT_MAX_ANGLE)
                        theta_d_y = func_clip(drift_y, -DRIFT_MAX_ANGLE, DRIFT_MAX_ANGLE)
                    else:
                        theta_d_x = 0.0
                        theta_d_y = 0.0
                else:
                    # Steering mode: WORLD FRAME control (joystick commands in world coordinates)
                    if abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS:
                        # Joystick commands in world frame
                        world_fwd = -js_fwd      # Forward in world frame
                        world_strafe = js_strafe # Right in world frame

                        # Rotate to body frame using yaw angle (theta_z)
                        # Body frame commands = Rotation matrix * World frame commands
                        cos_yaw = np.cos(theta_z)
                        sin_yaw = np.sin(theta_z)
                        body_fwd = cos_yaw * world_fwd - sin_yaw * world_strafe
                        body_strafe = sin_yaw * world_fwd + cos_yaw * world_strafe

                        # Convert to desired tilt angles
                        theta_d_x = THETA_MAX_RAD * body_strafe
                        theta_d_y = THETA_MAX_RAD * body_fwd
                    else:
                        theta_d_x = 0.0
                        theta_d_y = 0.0

                # Compute errors
                e_x = theta_d_x - theta_x
                e_y = theta_d_y - theta_y

                # Compute control torques
                if not abort:
                    # Balance controller (always active)
                    Tx_balance = pid_y.update(e_y, dt)  # pitch -> Tx
                    Ty_balance = pid_x.update(e_x, dt)  # roll -> Ty

                    if BALANCE_MODE:
                        # Pure balance with Tz turning
                        pid_x_steer.reset()
                        pid_y_steer.reset()
                        Tx = func_clip(Tx_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Ty = func_clip(Ty_balance, -PWM_USER_CAP, PWM_USER_CAP)
                        Tz = TZ_MAX * js_spin  # Spin control with left joystick
                    else:
                        # Balance + steering
                        if ((abs(js_fwd) > VEL_CMD_EPS or abs(js_strafe) > VEL_CMD_EPS) and
                            abs(theta_x) < TH_STEER_PRIOR_RAD and abs(theta_y) < TH_STEER_PRIOR_RAD):
                            Tx_steer = pid_y_steer.update(e_y, dt)
                            Ty_steer = pid_x_steer.update(e_x, dt)
                        else:
                            pid_x_steer.reset()
                            pid_y_steer.reset()
                            Tx_steer = 0.0
                            Ty_steer = 0.0

                        Tx = func_clip(Tx_balance + Tx_steer, -PWM_USER_CAP, PWM_USER_CAP)
                        Ty = func_clip(Ty_balance + Ty_steer, -PWM_USER_CAP, PWM_USER_CAP)
                        Tz = TZ_MAX * js_spin  # Direct spin control with left joystick
                else:
                    # Emergency stop
                    Tx = Ty = Tz = 0.0
                    pid_x.reset()
                    pid_y.reset()
                    pid_x_steer.reset()
                    pid_y_steer.reset()
                    print(f"ABORT: θx={np.rad2deg(theta_x):.1f}°, θy={np.rad2deg(theta_y):.1f}°")

                # ================================================================
                # SEND MOTOR COMMANDS
                # ================================================================
                u3, u1, u2 = calc_torque_conv(Tx, Ty, Tz)
                u1 = -func_clip(u1, -PWM_MAX, PWM_MAX)
                u2 = -func_clip(u2, -PWM_MAX, PWM_MAX)
                u3 = -func_clip(u3, -PWM_MAX, PWM_MAX)

                command.utime = int(time.time() * 1e6)
                command.pwm[0] = u1
                command.pwm[1] = u2
                command.pwm[2] = u3
                lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

                # ================================================================
                # DATA LOGGING
                # ================================================================
                # Get individual PID terms for logging
                p_x, i_x, d_x = pid_x.get_terms(e_x)
                p_y, i_y, d_y = pid_y.get_terms(e_y)

                data = [
                    i, t_now, Tx, Ty, Tz, u1, u2, u3,
                    np.rad2deg(theta_x), np.rad2deg(theta_y), np.rad2deg(theta_z),
                    np.rad2deg(theta_d_x), np.rad2deg(theta_d_y),
                    phi_x, phi_y, dphi_x, dphi_y,
                    psi_1, psi_2, psi_3, dpsi_1, dpsi_2, dpsi_3,
                    np.rad2deg(e_x), np.rad2deg(e_y), float(abort),
                    pid_x.kp, pid_x.ki, pid_x.kd, pid_y.kp, pid_y.ki, pid_y.kd,
                    p_x, i_x, d_x, p_y, i_y, d_y,
                ]
                dl.appendData(data)

                # ================================================================
                # CONSOLE OUTPUT
                # ================================================================
                abort_str = " [ABORT]" if abort else ""

                if BALANCE_MODE:
                    drift_str = "ON" if DRIFT_CORRECTION_ENABLED else "OFF"
                    print(f"[BAL] t={t_now:6.2f}s{abort_str} | "
                          f"θ=[{np.rad2deg(theta_x):+5.1f}°,{np.rad2deg(theta_y):+5.1f}°] "
                          f"d=[{np.rad2deg(theta_d_x):+4.1f}°,{np.rad2deg(theta_d_y):+4.1f}°] | "
                          f"φ=[{phi_x:+5.2f},{phi_y:+5.2f}] | Drift:{drift_str}")
                    print(f"      PID: Kp={pid_x.kp:.2f} Ki={pid_x.ki:.3f} Kd={pid_x.kd:.3f} | "
                          f"T=[{Tx:+5.2f},{Ty:+5.2f},{Tz:+5.2f}] | Tuning: {tune_names[cur_tune]}")
                else:
                    print(f"[STR] t={t_now:6.2f}s{abort_str} | "
                          f"θ=[{np.rad2deg(theta_x):+5.1f}°,{np.rad2deg(theta_y):+5.1f}°] "
                          f"d=[{np.rad2deg(theta_d_x):+5.1f}°,{np.rad2deg(theta_d_y):+5.1f}°] | "
                          f"js=[{js_fwd:+.2f},{js_strafe:+.2f}]")
                    print(f"      BAL Kp={pid_x.kp:.2f} Ki={pid_x.ki:.3f} | STR Kp={pid_x_steer.kp:.2f} Ki={pid_x_steer.ki:.3f} | "
                          f"T=[{Tx:+5.2f},{Ty:+5.2f},{Tz:+5.2f}] | Tuning: {tune_names[cur_tune]}")

            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt - stopping motors")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[:] = [0.0, 0.0, 0.0]
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

    finally:
        print(f"Saving data to {filename}...")
        dl.writeOut()
        listening = False
        print("Stopping LCM listener...")
        listener_thread.join(timeout=1)
        controller_thread.join(timeout=1)
        controller.on_options_press()
        print("Shutting down motors...")
        command = mbot_motor_pwm_t()
        command.utime = int(time.time() * 1e6)
        command.pwm[:] = [0.0, 0.0, 0.0]
        lc.publish("MBOT_MOTOR_PWM_CMD", command.encode())

if __name__ == "__main__":
    main()
