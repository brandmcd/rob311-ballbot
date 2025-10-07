"""
ROB 311 - Fall 2025
Author: Prof. Greg Formosa
University of Michigan

Script is meant to test PS4 Bluetooth controller signals, without additional overhead.
Will store data as a "test_BT_[#].txt" file that can be parsed in Matlab to show button signals.
Run script, choose a test number to store data as (will overwrite if file already exists), and then
press various buttons on the connected controller to see their signals.
Careful -- pressing "Options" will close the comms thread for that test run, and no more signals 
will be retrieved from the controller.

See "ps4_controller_api.py" for additional info on all button functions & values. Signals have been 
normalized to the range of [0,1].

Controller must first be connected over bluetooth (as a joystick device) on the js0 device kernel.

Press Ctrl+C to stop script at any time.

"""

import time
import lcm
import threading
import numpy as np
from DataLogger import dataLogger
from ps4_controller_api import PS4InputHandler

# Constants for the control loop
FREQ = 200  # Frequency of control loop in Hz
DT = 1 / FREQ  # Time step for each iteration in seconds

def main():
    # === Data Logging Initialization ===
    # Prompt user for trial number and create a data logger
    trial_num = int(input("Test Number? "))
    filename = f"test_BT_{trial_num}.txt"
    dl = dataLogger(filename)
    
    # === Controller Initialization ===
    # Create an instance of the PS4 controller handler
    controller = PS4InputHandler(interface="/dev/input/js0",connecting_using_ds4drv=False)
    # Start a separate thread to listen for controller inputs
    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.daemon = True  # Ensures the thread stops with the main program
    controller_thread.start()
    print("PS4 Controller is active. . .")

    try:
        # === Main Control Loop ===
        print("Starting loop...")
        time.sleep(DT)
        i = 0  # Iteration counter
        t_start = time.time()
        # Store variable names as header to data logged, for easier parsing in Matlab
        # TODO: update data header variables names to match actual data stored
        data = ["i t_now js_L_x js_L_y js_R_x js_R_y trigger_L2 trigger_R2 shoulder_L1 shoulder_R1 but_x but_cir but_tri but_sq dir_horz dir_vert"]
        dl.appendData(data)

        while True:
            time.sleep(DT)
            t_now = time.time() - t_start  # Elapsed time
            i += 1
            
            try:
                # retreive dictionary of button press signals from handler
                bt_signals = controller.get_signals()
                # parse out individual buttons you want data from
                js_L_x = bt_signals["js_L_x"]
                js_L_y = bt_signals["js_L_y"]
                js_R_x = bt_signals["js_R_x"]
                js_R_y = bt_signals["js_R_y"]
                trigger_L2 = bt_signals["trigger_L2"]
                trigger_R2 = bt_signals["trigger_R2"]
                shoulder_L1 = bt_signals["shoulder_L1"]
                shoulder_R1 = bt_signals["shoulder_R1"]
                but_x = bt_signals["but_x"]
                but_cir = bt_signals["but_cir"]
                but_tri = bt_signals["but_tri"]
                but_sq = bt_signals["but_sq"]
                dir_horz = bt_signals["dpad_horiz"]
                dir_vert = bt_signals["dpad_vert"] 

                # Store data in data logger
                # TODO: update variables to match data header names
                data = [i, t_now, js_L_x, js_L_y, js_R_x, js_R_y, trigger_L2, trigger_R2, shoulder_L1, shoulder_R1, but_x, but_cir, but_tri, but_sq, dir_horz, dir_vert]
                dl.appendData(data)
                # Print out data
                # TODO: update as necessary for what info you want to see in terminal
                print(
                    f"Time: {t_now:.3f}s | "
                    f"js_L_x: {js_L_x:.2f}, js_L_y: {js_L_y:.2f} | js_R_x: {js_R_x:.2f}, js_R_y: {js_R_y:.2f} | "
                    f"trigger_L2: {trigger_L2:.2f}, trigger_R2: {trigger_R2:.2f} | shoulder_L1: {shoulder_L1}, shoulder_R1: {shoulder_R1} | "
                    f"but_x: {but_x}, but_cir: {but_cir} | but_tri: {but_tri}, but_sq: {but_sq} | "
                    f"dir_horz: {dir_horz}, dir_vert: {dir_vert} | "
                )
            
            except KeyError:
                print("Waiting for sensor data...")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    
    finally:
        # Save/log data
        print(f"Saving data as {filename}...")
        dl.writeOut()  # Write logged data to the file
        # Stop Bluetooth thread
        controller_thread.join(timeout=1)  # Wait up to 1 second for thread to finish
        controller.on_options_press()

if __name__ == "__main__":
    main()