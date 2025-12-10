#!/usr/bin/env python3
"""
Script to plot PID contributions over time compared to error from log files.
Visualizes P, I, D terms and error for both X and Y axes.
"""

import numpy as np
import matplotlib
# Use Agg backend by default for headless operation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path


def load_log_file(filename):
    """Load and parse log file data."""
    try:
        # Read header
        with open(filename, 'r') as f:
            header = f.readline().strip().split()

        # Load data
        data = np.loadtxt(filename, skiprows=1)

        # Create dictionary mapping column names to data
        log_data = {}
        for i, col_name in enumerate(header):
            log_data[col_name] = data[:, i]

        return log_data
    except Exception as e:
        print(f"Error loading file {filename}: {e}")
        sys.exit(1)


def plot_pid_analysis(log_data, axis='x', filename=''):
    """
    Plot PID contributions and error over time for specified axis.

    Args:
        log_data: Dictionary containing log data
        axis: 'x' or 'y' - which axis to plot
        filename: Name of log file for title
    """
    axis = axis.lower()

    # Extract data
    time = log_data['t_now']
    error = log_data[f'e_{axis}']
    p_term = log_data[f'p_{axis}']
    i_term = log_data[f'i_{axis}']
    d_term = log_data[f'd_{axis}']

    # Calculate total control output
    u_total = p_term + i_term + d_term

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'PID Analysis - {axis.upper()} Axis\n{filename}', fontsize=14, fontweight='bold')

    # Plot 1: Error over time
    axes[0].plot(time, np.rad2deg(error), 'r-', linewidth=1.5, label='Error')
    axes[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('Error (degrees)', fontsize=11)
    axes[0].set_title('Error Over Time', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Plot 2: Individual PID contributions
    axes[1].plot(time, p_term, 'b-', linewidth=1.5, label='P term', alpha=0.8)
    axes[1].plot(time, i_term, 'g-', linewidth=1.5, label='I term', alpha=0.8)
    axes[1].plot(time, d_term, 'orange', linewidth=1.5, label='D term', alpha=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_ylabel('Control Output', fontsize=11)
    axes[1].set_title('PID Contributions Over Time', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Plot 3: Total output vs Error (comparison)
    ax3_1 = axes[2]
    ax3_2 = ax3_1.twinx()

    # Plot control output on left axis
    ax3_1.plot(time, u_total, 'purple', linewidth=1.5, label='Total Control Output', alpha=0.8)
    ax3_1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3_1.set_xlabel('Time (s)', fontsize=11)
    ax3_1.set_ylabel('Control Output', fontsize=11, color='purple')
    ax3_1.tick_params(axis='y', labelcolor='purple')
    ax3_1.grid(True, alpha=0.3)

    # Plot error on right axis
    ax3_2.plot(time, np.rad2deg(error), 'r-', linewidth=1.5, label='Error', alpha=0.6)
    ax3_2.set_ylabel('Error (degrees)', fontsize=11, color='r')
    ax3_2.tick_params(axis='y', labelcolor='r')

    axes[2].set_title('Total Output vs Error', fontsize=12)

    # Combine legends
    lines1, labels1 = ax3_1.get_legend_handles_labels()
    lines2, labels2 = ax3_2.get_legend_handles_labels()
    ax3_1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    return fig


def plot_combined_analysis(log_data, filename=''):
    """
    Plot combined analysis showing both X and Y axes and contribution breakdown.
    """
    time = log_data['t_now']

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Combined PID Analysis\n{filename}', fontsize=14, fontweight='bold')

    # Plot X axis error and contributions
    ax = axes[0, 0]
    ax.plot(time, np.rad2deg(log_data['e_x']), 'r-', linewidth=1.5, label='Error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Error (degrees)', fontsize=10)
    ax.set_title('X Axis - Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot Y axis error and contributions
    ax = axes[0, 1]
    ax.plot(time, np.rad2deg(log_data['e_y']), 'r-', linewidth=1.5, label='Error')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_ylabel('Error (degrees)', fontsize=10)
    ax.set_title('Y Axis - Error', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot X axis PID contributions
    ax = axes[1, 0]
    ax.plot(time, log_data['p_x'], 'b-', linewidth=1.5, label='P term', alpha=0.8)
    ax.plot(time, log_data['i_x'], 'g-', linewidth=1.5, label='I term', alpha=0.8)
    ax.plot(time, log_data['d_x'], 'orange', linewidth=1.5, label='D term', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Control Output', fontsize=10)
    ax.set_title('X Axis - PID Contributions', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot Y axis PID contributions
    ax = axes[1, 1]
    ax.plot(time, log_data['p_y'], 'b-', linewidth=1.5, label='P term', alpha=0.8)
    ax.plot(time, log_data['i_y'], 'g-', linewidth=1.5, label='I term', alpha=0.8)
    ax.plot(time, log_data['d_y'], 'orange', linewidth=1.5, label='D term', alpha=0.8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('Control Output', fontsize=10)
    ax.set_title('Y Axis - PID Contributions', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def print_statistics(log_data, axis='x'):
    """Print statistics about PID performance."""
    axis = axis.lower()
    error = log_data[f'e_{axis}']
    p_term = log_data[f'p_{axis}']
    i_term = log_data[f'i_{axis}']
    d_term = log_data[f'd_{axis}']

    print(f"\n{'='*60}")
    print(f"Statistics for {axis.upper()} Axis")
    print(f"{'='*60}")
    print(f"Time span: {log_data['t_now'][0]:.2f}s to {log_data['t_now'][-1]:.2f}s")
    print(f"Duration: {log_data['t_now'][-1] - log_data['t_now'][0]:.2f}s")
    print(f"Data points: {len(error)}")
    print(f"\nError Statistics (degrees):")
    print(f"  Mean:     {np.rad2deg(np.mean(error)):8.3f}")
    print(f"  Std Dev:  {np.rad2deg(np.std(error)):8.3f}")
    print(f"  RMS:      {np.rad2deg(np.sqrt(np.mean(error**2))):8.3f}")
    print(f"  Max:      {np.rad2deg(np.max(error)):8.3f}")
    print(f"  Min:      {np.rad2deg(np.min(error)):8.3f}")
    print(f"\nControl Output Statistics:")
    print(f"  P term - Mean: {np.mean(p_term):8.3f}, Max: {np.max(np.abs(p_term)):8.3f}")
    print(f"  I term - Mean: {np.mean(i_term):8.3f}, Max: {np.max(np.abs(i_term)):8.3f}")
    print(f"  D term - Mean: {np.mean(d_term):8.3f}, Max: {np.max(np.abs(d_term)):8.3f}")

    # Calculate contribution percentages
    total_contribution = np.abs(p_term) + np.abs(i_term) + np.abs(d_term)
    valid_idx = total_contribution > 1e-6  # Avoid division by zero

    if np.any(valid_idx):
        p_pct = np.mean(np.abs(p_term[valid_idx]) / total_contribution[valid_idx]) * 100
        i_pct = np.mean(np.abs(i_term[valid_idx]) / total_contribution[valid_idx]) * 100
        d_pct = np.mean(np.abs(d_term[valid_idx]) / total_contribution[valid_idx]) * 100

        print(f"\nAverage Contribution (%):")
        print(f"  P term: {p_pct:6.2f}%")
        print(f"  I term: {i_pct:6.2f}%")
        print(f"  D term: {d_pct:6.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Plot PID contributions over time from log files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s main_control_1.txt                    # Save both axes to main_control_1_pid_x_axis.png and main_control_1_pid_y_axis.png
  %(prog)s main_control_1.txt --axis x           # Save only X axis to main_control_1_pid.png
  %(prog)s main_control_1.txt --combined         # Save combined view to main_control_1_combined.png
  %(prog)s main_control_1.txt --save custom.png  # Save to custom filename
  %(prog)s main_control_1.txt --show             # Display plots interactively instead of saving
        """
    )

    parser.add_argument('logfile', type=str, help='Path to log file')
    parser.add_argument('--axis', type=str, choices=['x', 'y', 'both'], default='both',
                        help='Which axis to plot (default: both)')
    parser.add_argument('--combined', action='store_true',
                        help='Show combined analysis view')
    parser.add_argument('--stats', action='store_true',
                        help='Print statistics')
    parser.add_argument('--save', type=str, metavar='FILE',
                        help='Save plot to specific filename (default: auto-generate from logfile name)')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (default: save to PNG file)')

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.logfile).exists():
        print(f"Error: File '{args.logfile}' not found")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.logfile}...")
    log_data = load_log_file(args.logfile)
    print(f"Loaded {len(log_data['t_now'])} data points")

    # Print statistics if requested
    if args.stats or args.axis != 'both':
        if args.axis in ['x', 'both']:
            print_statistics(log_data, 'x')
        if args.axis in ['y', 'both']:
            print_statistics(log_data, 'y')

    # Create plots
    figures = []

    if args.combined:
        fig = plot_combined_analysis(log_data, args.logfile)
        figures.append(fig)
    else:
        if args.axis in ['x', 'both']:
            fig = plot_pid_analysis(log_data, 'x', args.logfile)
            figures.append(fig)
        if args.axis in ['y', 'both']:
            fig = plot_pid_analysis(log_data, 'y', args.logfile)
            figures.append(fig)

    # Save or show plots
    if args.save:
        save_filename = args.save
    else:
        # Auto-generate filename from logfile name
        base_name = Path(args.logfile).stem
        if args.combined:
            save_filename = f"{base_name}_combined.png"
        else:
            save_filename = f"{base_name}_pid.png"

    if len(figures) == 1:
        figures[0].savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_filename}")
    else:
        # Save multiple figures with suffixes
        base_name = Path(save_filename).stem
        extension = Path(save_filename).suffix
        for i, (fig, name) in enumerate(zip(figures, ['x_axis', 'y_axis'])):
            filename = f"{base_name}_{name}{extension}"
            fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {filename}")

    if args.show:
        # Only show if explicitly requested
        plt.show()


if __name__ == '__main__':
    main()
