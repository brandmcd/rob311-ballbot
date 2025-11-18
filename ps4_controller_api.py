"""
ROB 311 - Fall 2025
Author: Prof. Greg Formosa & GSI Yilin Ma
University of Michigan

PS4 Input Handler API - from https://pypi.org/project/pyPS4Controller/
Import this class into your control/test files to interface with your BT controller.

This module provides the `PS4InputHandler` class to interface with a PS4 controller. 
It captures inputs from the left and right Joysticks, Triggers, and Shoulder buttons, 
D-Pad buttons, and X/Circle/Triangle/Square buttons, and processes them into normalized 
values suitable for use in robotics or control systems (generally [0,1]).

Continuous inputs (joysticks and triggers) are normalized to a range of [0.0, 1.0], 
while all other buttons are designated as 1 (pressed) or 0 (released).
The "Options" button is specifically designated as an emergency stop to turn off the 
controller thread, so pressing it will shutdown any signals sent from the controller.

All signals are stored in a dictionary accessible via the `get_signals` method.

Example Usage:
1. Initialize the `PS4InputHandler` with the controller's interface:
    handler = PS4InputHandler(interface="/dev/input/js0")
2. Start the controller's event listener:
    handler.listen(timeout=10)
3. Retrieve real-time control signals:
    signals = handler.get_signals()

"""

import sys
import numpy as np
from pyPS4Controller.controller import Controller

# Scale factor for normalizing joystick and trigger values
JOYSTICK_SCALE = 32767

class PS4InputHandler(Controller):
    def __init__(self, interface, **kwargs):
        super().__init__(interface, **kwargs)
        self.signals = {
            "js_L_x": 0.0,
            "js_L_y": 0.0,
            "js_R_x": 0.0,
            "js_R_y": 0.0,
            "trigger_L2": 0.0,
            "trigger_R2": 0.0,
            "shoulder_L1": 0,
            "shoulder_R1": 0,
            "but_x": 0,
            "but_cir": 0,
            "but_tri": 0,
            "but_sq": 0,
           "dpad_horiz":0,
           "dpad_vert":0,
        }

    # === Left Joystick (L3) ===
    def on_L3_left(self, value):
        self.signals["js_L_x"] = value / JOYSTICK_SCALE
    def on_L3_right(self, value):
        self.signals["js_L_x"] = value / JOYSTICK_SCALE
    def on_L3_up(self, value):
        self.signals["js_L_y"] = value / JOYSTICK_SCALE
    def on_L3_down(self, value):
        self.signals["js_L_y"] = value / JOYSTICK_SCALE
    def on_L3_x_at_rest(self):
        self.signals["js_L_x"] = 0.0
    def on_L3_y_at_rest(self):
        self.signals["js_L_y"] = 0.0

    # === Right Joystick (R3) ===
    def on_R3_left(self, value):
        self.signals["js_R_x"] = value / JOYSTICK_SCALE
    def on_R3_right(self, value):
        self.signals["js_R_x"] = value / JOYSTICK_SCALE
    def on_R3_up(self, value):
        self.signals["js_R_y"] = value / JOYSTICK_SCALE
    def on_R3_down(self, value):
        self.signals["js_R_y"] = value / JOYSTICK_SCALE
    def on_R3_x_at_rest(self):
        self.signals["js_R_x"] = 0.0
    def on_R3_y_at_rest(self):
        self.signals["js_R_y"] = 0.0
        
    # === Triggers (L2 and R2) ===
    def on_L2_press(self, value):
        self.signals["trigger_L2"] = (value + JOYSTICK_SCALE) / (2 * JOYSTICK_SCALE)
    def on_L2_release(self):
        self.signals["trigger_L2"] = 0.0
    def on_R2_press(self, value):
        self.signals["trigger_R2"] = (value + JOYSTICK_SCALE) / (2 * JOYSTICK_SCALE)
    def on_R2_release(self):
        self.signals["trigger_R2"] = 0.0

    # === Shoulder Buttons (L1 and R1) ===
    def on_L1_press(self):
        self.signals["shoulder_L1"] = 1
    def on_L1_release(self):
        self.signals["shoulder_L1"] = 0   
    def on_R1_press(self):
        self.signals["shoulder_R1"] = 1
    def on_R1_release(self):
        self.signals["shoulder_R1"] = 0

    # === Buttons ===
    def on_x_press(self):
        self.signals["but_x"] = 1
    def on_x_release(self):
        self.signals["but_x"] = 0
    def on_triangle_press(self):
        self.signals["but_tri"] = 1
    def on_triangle_release(self):
        self.signals["but_tri"] = 0
    def on_circle_press(self):
        self.signals["but_cir"] = 1
    def on_circle_release(self):
        self.signals["but_cir"] = 0
    def on_square_press(self):
        self.signals["but_sq"] = 1
    def on_square_release(self):
        self.signals["but_sq"] = 0

    # === D-Pad ===
    # TODO: update code for D-Pad Buttons to increase/decrease value when pressed (not to toggle 0 or 1)
    # Hint: you may not need the release functions (that will reset your values...)
    def on_left_arrow_press(self):
            self.signals["dpad_horiz"] -= 0.1
    def on_right_arrow_press(self):
            self.signals["dpad_horiz"] += 0.1
    # def on_left_right_arrow_release(self):
    #     self.signals["dir_L"] = 0
    #     self.signals["dir_R"] = 0
    def on_up_arrow_press(self):
            self.signals["dpad_vert"] += 0.1
    def on_down_arrow_press(self):
            self.signals["dpad_vert"] -= 0.1
    # def on_up_down_arrow_release(self):
    #     self.signals["dir_U"] = 0
    #     self.signals["dir_D"] = 0

    # === Options Button ===
    def on_options_press(self):
        # Exit the PS4 controller thread gracefully.
        print("Exiting PS4 controller thread.")
        # sys.exit()

    # === Method to Call Signals ===
    def get_signals(self):
        # Get the current state of all control signals.
        # return: A dictionary containing the current values of all signals.
        return self.signals

