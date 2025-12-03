import time
import threading

import lcm
import numpy as np

from mbot_lcm_msgs.mbot_motor_pwm_t import mbot_motor_pwm_t
from mbot_lcm_msgs.mbot_balbot_feedback_t import mbot_balbot_feedback_t

from DataLogger3 import dataLogger
from ps4_controller_api import PS4InputHandler
from pid import NEW_PID, PID

# ==============================
#  GLOBAL CONSTANTS / GEOMETRY
# ==============================

FREQ = 100.0                         # control loop frequency [Hz]
DT = 1.0 / FREQ

# motor / encoder / geometry (ROB 311 default kit)
PWM_MAX = 0.96
PWM_USER_CAP = 0.96                  # additional cap for safety during control

N_GEARBOX = 70                       # gear ratio
N_ENC = 64                           # encoder ticks / rev (per motor shaft)

R_W = 0.048                          # omni wheel radius [m]
R_K = 0.121                          # ball radius [m]
ALPHA = np.deg2rad(45.0)             # wheel tilt angle [rad]

# balance / steering configuration
TH_ABORT_DEG = 25.0                  # hard abort if tilt exceeds this (deg)
TH_ABORT_RAD = np.deg2rad(TH_ABORT_DEG)

THETA_MAX_DEG = 3.0                  # nominal max commanded lean [deg]
THETA_MAX_RAD = np.deg2rad(THETA_MAX_DEG)

# we will saturate even harder in the cascaded controller
THETA_HARD_MAX_RAD = np.deg2rad(2.0)  # keep lean very small for this small/tippy bot

VEL_MAX = 1.0                        # max commanded ball speed (approx rad/s equiv)
VEL_FILT_ALPHA = 0.3                 # EMA for ball velocity
THETA_FILT_ALPHA = 0.2               # EMA for IMU angles

DEADZONE = 0.08                      # joystick deadzone (unitless, 0..1)
VEL_CMD_EPS = 1e-3                   # threshold to treat joystick as zero

# LCM globals
listening = False
msg_feedback = mbot_balbot_feedback_t()
last_feedback_time = 0.0


# ======================
#   HELPER FUNCTIONS
# ======================

def func_clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def too_lean(theta_x, theta_y):
    """Return True if the robot is leaned beyond TH_ABORT_RAD."""
    return (abs(theta_x) > TH_ABORT_RAD) or (abs(theta_y) > TH_ABORT_RAD)


def calc_enc2rad(ticks: float) -> float:
    """Convert encoder ticks at the wheel motor to radians at the wheel surface."""
    return (2.0 * np.pi * ticks) / (N_ENC * N_GEARBOX)


def calc_torque_conv(Tx: float, Ty: float, Tz: float):
    """
    Convert desired ball torques (Tx, Ty, Tz) in body frame into
    per-wheel torque commands u1..u3.

    This is the standard inverse of the kinematic Jacobian for three
    omni wheels placed every 120 degrees around the ball, tilted by ALPHA.
    """
    ca = np.cos(ALPHA)

    u1 = (1.0 / 3.0) * (Tz - (2.0 * Ty) / ca)
    u2 = (1.0 / 3.0) * (Tz + (1.0 / ca) * (-np.sqrt(3.0) * Tx + Ty))
    u3 = (1.0 / 3.0) * (Tz + (1.0 / ca) * ( np.sqrt(3.0) * Tx + Ty))
    return u1, u2, u3


def calc_kinematic_conv(psi1: float, psi2: float, psi3: float):
    """
    Map wheel rotation (psi1..psi3) into equivalent ball rotation
    (phi_x, phi_y, phi_z) in body coordinates.

    The scaling matches the kit geometry (ROB 311 notes).
    """
    # These expressions match the staff-provided template
    phi_x = np.sqrt(2.0 / 3.0) * (R_W / R_K) * (psi2 - psi3)
    phi_y = np.sqrt(2.0) / 3.0 * (R_W / R_K) * (-2.0 * psi1 + psi2 + psi3)
    phi_z = np.sqrt(2.0) / 3.0 * (R_W / R_K) * (psi1 + psi2 + psi3)
    return phi_x, phi_y, phi_z


# ======================
#   LCM CALLBACKS
# ======================

def feedback_handler(channel, data):
    global msg_feedback, last_feedback_time
    msg_feedback = mbot_balbot_feedback_t.decode(data)
    last_feedback_time = time.time()


def lcm_listener(lc):
    global listening
    while listening:
        try:
            lc.handle_timeout(100)
        except Exception as e:
            print(f"[LCM] listening error: {e}")
            break


# ======================
#     MAIN CONTROL
# ======================

def main():
    global listening

    # --------------
    # Logging setup
    # --------------
    trial_num = int(input("Test number? "))
    filename = f"main_control_{trial_num}.txt"
    dl = dataLogger(filename)

    # --------------
    # LCM setup
    # --------------
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    lc.subscribe("MBOT_BALBOT_FEEDBACK", feedback_handler)
    listening = True
    listener_thread = threading.Thread(target=lcm_listener, args=(lc,), daemon=True)
    listener_thread.start()
    print("[INIT] LCM listener started")

    # --------------
    # PS4 controller
    # --------------
    controller = PS4InputHandler(interface="/dev/input/js0", connecting_using_ds4drv=False)
    controller_thread = threading.Thread(target=controller.listen, args=(10,), daemon=True)
    controller_thread.start()
    print("[INIT] PS4 controller thread started")
    print("       Triangle = kill, Circle = re-zero IMU, Cross = toggle steer mode")

    # --------------
    # PIDs
    # --------------
    # Inner loop: angle (θ) control → ball torques
    pid_theta_x = NEW_PID(
        kp=8.6, ki=0, kd=0.02,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=-0.3, integ_max=0.3,
        d_window=5,
    )
    pid_theta_y = NEW_PID(
        kp=8.6, ki=0, kd=0.02,
        u_min=-PWM_USER_CAP, u_max=PWM_USER_CAP,
        integ_min=-0.3, integ_max=0.3,
        d_window=5,
    )

    # Outer loop: ball velocity → desired lean angle
    pid_vel_x = NEW_PID(
        kp=0.8, ki=0.4, kd=0.0,
        u_min=-THETA_MAX_RAD, u_max=THETA_MAX_RAD,
        integ_min=-THETA_MAX_RAD, integ_max=THETA_MAX_RAD,
        d_window=1,
    )
    pid_vel_y = NEW_PID(
        kp=0.8, ki=0.4, kd=0.0,
        u_min=-THETA_MAX_RAD, u_max=THETA_MAX_RAD,
        integ_min=-THETA_MAX_RAD, integ_max=THETA_MAX_RAD,
        d_window=1,
    )

    # --------------
    # State & bias
    # --------------
    theta_x = 0.0
    theta_y = 0.0
    theta_z = 0.0

    theta_x_bias = 0.0
    theta_y_bias = 0.0
    theta_z_bias = 0.0

    vx_filt = 0.0
    vy_filt = 0.0

    # ball pose (approx) from wheel encoders
    phi_x = phi_y = phi_z = 0.0

    # steering enable toggle
    steer_enabled = True
    prev_cross = False
    prev_circle = False

    # encoder offsets for zero
    enc_pos_1_start = 0
    enc_pos_2_start = 0
    enc_pos_3_start = 0
    encoder_zeroed = False

    # --------------
    # Logging header
    # --------------
    header = [
        "i",
        "t",
        "Tx", "Ty", "Tz",
        "u1", "u2", "u3",
        "theta_x_deg", "theta_y_deg",
        "theta_dx_deg", "theta_dy_deg",
        "vx", "vy", "vx_cmd", "vy_cmd",
        "abort",
        "kp_theta", "ki_theta", "kd_theta",
        "kp_vel", "ki_vel",
    ]
    dl.appendData(header)

    # --------------
    # Control loop
    # --------------
    i = 0
    t_start = time.time()
    prev_loop_time = t_start

    try:
        while True:
            loop_start = time.time()
            t_now = loop_start - t_start

            # Require fresh feedback before doing anything
            if last_feedback_time == 0.0 or (loop_start - last_feedback_time) > 0.5:
                # No fresh feedback: stop motors
                cmd = mbot_motor_pwm_t()
                cmd.utime = int(loop_start * 1e6)
                cmd.pwm[0] = 0.0
                cmd.pwm[1] = 0.0
                cmd.pwm[2] = 0.0
                lc.publish("MBOT_MOTOR_PWM_CMD", cmd.encode())
                time.sleep(0.01)
                continue

            # --- Read controller ---
            if controller.triangle:   # hard kill
                print("[CTRL] Triangle pressed → stopping")
                break

            # toggle steering mode on rising edge of Cross
            if controller.cross and not prev_cross:
                steer_enabled = not steer_enabled
                mode_str = "STEER+BALANCE" if steer_enabled else "BALANCE ONLY"
                print(f"[CTRL] Cross toggled → {mode_str}")
            prev_cross = controller.cross

            # Re-zero IMU on Circle rising edge
            if controller.circle and not prev_circle:
                theta_x_bias = msg_feedback.imu_angles_rpy[0]
                theta_y_bias = msg_feedback.imu_angles_rpy[1]
                theta_z_bias = msg_feedback.imu_angles_rpy[2]
                print("[CTRL] Circle pressed → IMU zeroed")
            prev_circle = controller.circle

            # --- Extract feedback ---
            # IMU orientation
            theta_x_raw = msg_feedback.imu_angles_rpy[0] - theta_x_bias
            theta_y_raw = msg_feedback.imu_angles_rpy[1] - theta_y_bias
            theta_z_raw = msg_feedback.imu_angles_rpy[2] - theta_z_bias

            theta_x = (1.0 - THETA_FILT_ALPHA) * theta_x + THETA_FILT_ALPHA * theta_x_raw
            theta_y = (1.0 - THETA_FILT_ALPHA) * theta_y + THETA_FILT_ALPHA * theta_y_raw
            theta_z = theta_z_raw  # yaw not filtered/used for now

            # encoder positions / velocities
            if not encoder_zeroed:
                enc_pos_1_start = msg_feedback.enc_ticks[0]
                enc_pos_2_start = msg_feedback.enc_ticks[1]
                enc_pos_3_start = msg_feedback.enc_ticks[2]
                encoder_zeroed = True

            enc_pos_1 = msg_feedback.enc_ticks[0] - enc_pos_1_start
            enc_pos_2 = msg_feedback.enc_ticks[1] - enc_pos_2_start
            enc_pos_3 = msg_feedback.enc_ticks[2] - enc_pos_3_start

            enc_dtick_1 = msg_feedback.enc_delta_ticks[0]
            enc_dtick_2 = msg_feedback.enc_delta_ticks[1]
            enc_dtick_3 = msg_feedback.enc_delta_ticks[2]
            enc_dt_us = max(1, msg_feedback.enc_delta_time)   # [us]

            psi_1 = calc_enc2rad(enc_pos_1)
            psi_2 = calc_enc2rad(enc_pos_2)
            psi_3 = calc_enc2rad(enc_pos_3)

            dpsi_1 = calc_enc2rad(enc_dtick_1) / (enc_dt_us * 1e-6)
            dpsi_2 = calc_enc2rad(enc_dtick_2) / (enc_dt_us * 1e-6)
            dpsi_3 = calc_enc2rad(enc_dtick_3) / (enc_dt_us * 1e-6)

            phi_x, phi_y, phi_z = calc_kinematic_conv(psi_1, psi_2, psi_3)
            dphi_x, dphi_y, dphi_z = calc_kinematic_conv(dpsi_1, dpsi_2, dpsi_3)

            # Filter velocities for outer loop
            vx_filt = (1.0 - VEL_FILT_ALPHA) * vx_filt + VEL_FILT_ALPHA * dphi_x
            vy_filt = (1.0 - VEL_FILT_ALPHA) * vy_filt + VEL_FILT_ALPHA * dphi_y

            # Safety check
            abort = too_lean(theta_x, theta_y)

            # dt for PIDs
            now_t = loop_start
            dt = max(1e-4, now_t - prev_loop_time)
            prev_loop_time = now_t

            # ------------------------------
            #  Joystick → desired velocities
            # ------------------------------
            # We'll use right stick for motion
            js_fwd = -controller.ry  # forward/back
            js_str = controller.rx   # left/right

            # deadzone
            js_fwd = js_fwd if abs(js_fwd) > DEADZONE else 0.0
            js_str = js_str if abs(js_str) > DEADZONE else 0.0

            # gentle shaping (square but preserve sign) for fine control near zero
            js_fwd = np.sign(js_fwd) * (js_fwd ** 2)
            js_str = np.sign(js_str) * (js_str ** 2)

            if steer_enabled:
                v_des_y = VEL_MAX * js_fwd
                v_des_x = VEL_MAX * js_str
            else:
                v_des_x = 0.0
                v_des_y = 0.0

            # ---------------
            #  Outer loop: v → θ_d
            # ---------------
            e_vx = v_des_x - vx_filt
            e_vy = v_des_y - vy_filt

            theta_d_x = pid_vel_x.update(e_vx, dt)
            theta_d_y = pid_vel_y.update(e_vy, dt)

            # Hard clamp desired angle for safety
            theta_d_x = func_clip(theta_d_x, -THETA_HARD_MAX_RAD, THETA_HARD_MAX_RAD)
            theta_d_y = func_clip(theta_d_y, -THETA_HARD_MAX_RAD, THETA_HARD_MAX_RAD)

            # ---------------
            #  Inner loop: θ → T
            # ---------------
            if abort:
                Tx = Ty = Tz = 0.0
                pid_theta_x.reset()
                pid_theta_y.reset()
                pid_vel_x.reset()
                pid_vel_y.reset()
            else:
                e_theta_x = theta_d_x - theta_x
                e_theta_y = theta_d_y - theta_y

                # mapping: pitch (y) → Tx, roll (x) → Ty
                Tx = pid_theta_y.update(e_theta_y, dt)   # pitch
                Ty = pid_theta_x.update(e_theta_x, dt)   # roll
                Tz = 0.0

                Tx = func_clip(Tx, -PWM_USER_CAP, PWM_USER_CAP)
                Ty = func_clip(Ty, -PWM_USER_CAP, PWM_USER_CAP)

            # ---------------
            #  Torque → wheel PWM
            # ---------------
            u1, u2, u3 = calc_torque_conv(Tx, Ty, Tz)
            u1 = func_clip(u1, -PWM_MAX, PWM_MAX)
            u2 = func_clip(u2, -PWM_MAX, PWM_MAX)
            u3 = func_clip(u3, -PWM_MAX, PWM_MAX)

            cmd = mbot_motor_pwm_t()
            cmd.utime = int(loop_start * 1e6)
            cmd.pwm[0] = float(u1)
            cmd.pwm[1] = float(u2)
            cmd.pwm[2] = float(u3)
            lc.publish("MBOT_MOTOR_PWM_CMD", cmd.encode())

            # ---------------
            #  Logging
            # ---------------
            row = [
                i,
                float(t_now),
                float(Tx), float(Ty), float(Tz),
                float(u1), float(u2), float(u3),
                float(np.rad2deg(theta_x)),
                float(np.rad2deg(theta_y)),
                float(np.rad2deg(theta_d_x)),
                float(np.rad2deg(theta_d_y)),
                float(vx_filt),
                float(vy_filt),
                float(v_des_x),
                float(v_des_y),
                int(abort),
                float(pid_theta_x.kp),
                float(pid_theta_x.ki),
                float(pid_theta_x.kd),
                float(pid_vel_x.kp),
                float(pid_vel_x.ki),
            ]
            dl.appendData(row)

            # maintain loop rate
            loop_end = time.time()
            elapsed = loop_end - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # periodic console print
            if i % 50 == 0:
                mode = "STEER" if steer_enabled else "BAL"
                print(
                    f"[{mode}] t={t_now:5.2f}s | "
                    f"θx={np.rad2deg(theta_x):+6.2f}° (d={np.rad2deg(theta_d_x):+5.2f}°) | "
                    f"θy={np.rad2deg(theta_y):+6.2f}° (d={np.rad2deg(theta_d_y):+5.2f}°) | "
                    f"vx={vx_filt:+5.3f}, vy={vy_filt:+5.3f} | "
                    f"v_des_x={v_des_x:+5.3f}, v_des_y={v_des_y:+5.3f} | "
                    f"u=[{u1:+5.2f},{u2:+5.2f},{u3:+5.2f}]"
                )

            i += 1

    except KeyboardInterrupt:
        print("\n[MAIN] KeyboardInterrupt → stopping")

    finally:
        # stop listener
        listening = False
        time.sleep(0.1)
        print("[MAIN] Shutting down motors and threads")

        # stop motors
        cmd = mbot_motor_pwm_t()
        cmd.utime = int(time.time() * 1e6)
        cmd.pwm[0] = 0.0
        cmd.pwm[1] = 0.0
        cmd.pwm[2] = 0.0
        lc.publish("MBOT_MOTOR_PWM_CMD", cmd.encode())

        # flush logger to disk
        dl.writeOut()

        # let threads wind down
        controller.on_options_press()
        listener_thread.join(timeout=1.0)
        controller_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
