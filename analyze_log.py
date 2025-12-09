#!/usr/bin/env python3
"""Quick script to analyze control logs"""
import pandas as pd
import numpy as np

# Read the log file
log_file = "main_control_1.txt"
print(f"Reading {log_file}...")

df = pd.read_csv(log_file, sep='\s+')

print(f"\nLog statistics:")
print(f"Total samples: {len(df)}")
print(f"Duration: {df['t_now'].max():.2f} seconds")
print(f"Sampling rate: {len(df)/df['t_now'].max():.1f} Hz")

# Look at balance performance
print(f"\n=== BALANCE PERFORMANCE ===")
print(f"Theta_x (deg): mean={df['theta_x'].mean():.2f}, std={df['theta_x'].std():.2f}, max={df['theta_x'].abs().max():.2f}")
print(f"Theta_y (deg): mean={df['theta_y'].mean():.2f}, std={df['theta_y'].std():.2f}, max={df['theta_y'].abs().max():.2f}")

# Look at control outputs
print(f"\n=== CONTROL OUTPUTS ===")
print(f"Tx: mean={df['Tx'].mean():.3f}, std={df['Tx'].std():.3f}, max={df['Tx'].abs().max():.3f}")
print(f"Ty: mean={df['Ty'].mean():.3f}, std={df['Ty'].std():.3f}, max={df['Ty'].abs().max():.3f}")

# Look at motor commands
print(f"\n=== MOTOR COMMANDS ===")
print(f"u1: mean={df['u1'].mean():.3f}, std={df['u1'].std():.3f}, max={df['u1'].abs().max():.3f}")
print(f"u2: mean={df['u2'].mean():.3f}, std={df['u2'].std():.3f}, max={df['u2'].abs().max():.3f}")
print(f"u3: mean={df['u3'].mean():.3f}, std={df['u3'].std():.3f}, max={df['u3'].abs().max():.3f}")

# Look at position drift
print(f"\n=== POSITION (DRIFT) ===")
print(f"phi_x: mean={df['phi_x'].mean():.3f}, std={df['phi_x'].std():.3f}, max={df['phi_x'].abs().max():.3f}")
print(f"phi_y: mean={df['phi_y'].mean():.3f}, std={df['phi_y'].std():.3f}, max={df['phi_y'].abs().max():.3f}")

# Look at position velocity
print(f"\n=== VELOCITY (DRIFT RATE) ===")
print(f"dphi_x: mean={df['dphi_x'].mean():.3f}, std={df['dphi_x'].std():.3f}, max={df['dphi_x'].abs().max():.3f}")
print(f"dphi_y: mean={df['dphi_y'].mean():.3f}, std={df['dphi_y'].std():.3f}, max={df['dphi_y'].abs().max():.3f}")

# Look at PID gains over time
print(f"\n=== PID GAINS (at end) ===")
print(f"Kp: {df['kp_x'].iloc[-1]:.2f}")
print(f"Ki: {df['ki_x'].iloc[-1]:.3f}")
print(f"Kd: {df['kd_x'].iloc[-1]:.3f}")

# Check for jittering - high frequency oscillations
print(f"\n=== JITTER ANALYSIS ===")
# Calculate angle changes per sample
theta_x_diff = df['theta_x'].diff().abs()
theta_y_diff = df['theta_y'].diff().abs()
print(f"Avg angle change per sample: theta_x={theta_x_diff.mean():.3f}°, theta_y={theta_y_diff.mean():.3f}°")
print(f"Max angle change per sample: theta_x={theta_x_diff.max():.3f}°, theta_y={theta_y_diff.max():.3f}°")

# Check control signal changes (jitter in control)
tx_diff = df['Tx'].diff().abs()
ty_diff = df['Ty'].diff().abs()
print(f"Avg control change per sample: Tx={tx_diff.mean():.3f}, Ty={ty_diff.mean():.3f}")
print(f"Max control change per sample: Tx={tx_diff.max():.3f}, Ty={ty_diff.max():.3f}")

# Show first 10 samples
print(f"\n=== FIRST 10 SAMPLES ===")
cols_to_show = ['t_now', 'theta_x', 'theta_y', 'theta_d_x', 'theta_d_y', 'Tx', 'Ty', 'phi_x', 'phi_y']
print(df[cols_to_show].head(10).to_string())

# Show last 10 samples
print(f"\n=== LAST 10 SAMPLES ===")
print(df[cols_to_show].tail(10).to_string())
