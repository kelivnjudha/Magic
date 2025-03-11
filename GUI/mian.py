import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
import keyboard
import time
import random
from mss import mss
from vncdotool import api
import colorsys
import math
from queue import Queue
import json
import sys

# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return (int(h * 180), int(s * 255), int(v * 255))

# Global variables
curr_x, curr_y = 960, 540  # Center of 1920x1080 screen
running = True
assist_enabled = False
assist_range = 500
mouse_speed = 1
assist_delay = 150
last_detected_time = 0
last_send_time = 0
screen_frame = None
screen_lock = threading.Lock()
vnc_connected = False
target_x, target_y = None, None
frame_queue = Queue(maxsize=1)
mask_queue = Queue(maxsize=1)
show_mask = True  # Always active for debugging
stop_event = threading.Event()
last_target_x, last_target_y = None, None
last_move_time = time.time()  # Initialize last_move_time

# Capture region parameters
capture_left = 760
capture_top = 428
capture_width = 400
capture_height = 225

# Setup VNC to Main PC
try:
    vnc = api.connect("127.0.0.1", password="200210")
    vnc_connected = True
    print("VNC Server connected successfully to 127.0.0.1:5900")
except Exception as e:
    print(f"Failed to connect to VNC: {e}")
    print("Please ensure TightVNC Server is running and configured correctly.")
    assist_enabled = False

# GUI control functions
def update_range(val):
    global assist_range
    assist_range = min(200, int(float(val)))  # Cap at 200px
    range_label.config(text=f"Range Radius: {assist_range} px")

def update_speed(val):
    global mouse_speed
    mouse_speed = int(float(val))
    speed_label.config(text=f"Mouse Speed: {mouse_speed}")

def update_assist_delay(val):
    global assist_delay
    assist_delay = int(float(val))
    delay_label.config(text=f"Reaction Delay: {assist_delay} ms")

def toggle_assist():
    global assist_enabled, last_detected_time, last_target_x, last_target_y
    assist_enabled = assist_var.get()
    status_label.config(text=f"Assist: {'Enabled' if assist_enabled else 'Disabled'} | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
    if assist_enabled:
        print("Aim assist enabled. Resetting target tracking.")
        last_detected_time = 0
        last_target_x, last_target_y = None, None
    else:
        print("Aim assist disabled.")

# Settings save/load functions
def save_settings():
    settings = {
        "hsv": {
            "lower_h": lower_h.get(),
            "upper_h": upper_h.get(),
            "lower_s": lower_s.get(),
            "upper_s": upper_s.get(),
            "lower_v": lower_v.get(),
            "upper_v": upper_v.get()
        }
    }
    try:
        with open("settings.json", "w") as f:
            json.dump(settings, f, indent=4)
        print("Settings saved to settings.json")
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings():
    try:
        with open("settings.json", "r") as f:
            settings = json.load(f)
        lower_h.set(settings["hsv"]["lower_h"])
        upper_h.set(settings["hsv"]["upper_h"])
        lower_s.set(settings["hsv"]["lower_s"])
        upper_s.set(settings["hsv"]["upper_s"])
        lower_v.set(settings["hsv"]["lower_v"])
        upper_v.set(settings["hsv"]["upper_v"])
        update_color_label(selected_color.get())
        print("Settings loaded from settings.json")
    except Exception as e:
        print(f"Error loading settings: {e}. Using default values.")

def set_pre_tuned_color(color):
    if color == "Red":
        lower_h.set(151) 
        upper_h.set(179)
        lower_s.set(163)
        upper_s.set(255)
        lower_v.set(58)
        upper_v.set(222)
    elif color == "Purple":
        lower_h.set(151)
        upper_h.set(179)
        lower_s.set(163)
        upper_s.set(255)
        lower_v.set(58)
        upper_v.set(222)
    elif color == "Yellow":
        lower_h.set(30)
        upper_h.set(30)
        lower_s.set(120)
        upper_s.set(255)
        lower_v.set(150)
        upper_v.set(255)
    update_color_label(color)
    print(f"Set pre-tuned color: {color}")

def update_color_label(color):
    color_label.config(text=f"Target Color: {color}")

# GUI Setup
root = tk.Tk()
icon = tk.PhotoImage(file='logo.png')
root.iconphoto(False, icon)
root.title("Magic Assist Beta v1 (Main PC) - Python 3.10.11")
root.geometry("800x700")
print("Tkinter root initialized.")

# Canvas for screen capture (scaled to 640x360, 16:9)
canvas = tk.Canvas(root, width=640, height=360)
canvas.pack()

# Control Frame
control_frame = ttk.Frame(root)
control_frame.pack(pady=10)

# Enable/Disable Toggle
assist_var = tk.BooleanVar(value=False)
ttk.Checkbutton(control_frame, text="Enable Aim Assist", variable=assist_var, command=toggle_assist).pack(side=tk.LEFT, padx=5)

# Range Slider
range_label = ttk.Label(control_frame, text=f"Range Radius: {assist_range} px")
range_label.pack(side=tk.LEFT, padx=5)
range_slider = ttk.Scale(control_frame, from_=20, to=200, orient=tk.HORIZONTAL, command=update_range)
range_slider.pack(side=tk.LEFT, padx=5)
range_slider.set(assist_range)

# Mouse Speed Slider
speed_label = ttk.Label(control_frame, text=f"Mouse Speed: {mouse_speed}")
speed_label.pack(side=tk.LEFT, padx=5)
speed_slider = ttk.Scale(control_frame, from_=1, to=200, orient=tk.HORIZONTAL, command=update_speed)
speed_slider.pack(side=tk.LEFT, padx=5)
speed_slider.set(mouse_speed)

# Reaction Delay Slider
delay_label = ttk.Label(control_frame, text=f"Reaction Delay: {assist_delay} ms")
delay_label.pack(side=tk.LEFT, padx=5)
delay_slider = ttk.Scale(control_frame, from_=0, to=500, orient=tk.HORIZONTAL, command=update_assist_delay)
delay_slider.pack(side=tk.LEFT, padx=5)
delay_slider.set(assist_delay)

# Color Selection and Input Frame
color_frame = ttk.Frame(root)
color_frame.pack(pady=5)

# Pre-tuned color dropdown
color_options = ["Red", "Purple", "Yellow"]
selected_color = tk.StringVar(value="Red")
ttk.OptionMenu(color_frame, selected_color, "Red", *color_options, command=lambda value: set_pre_tuned_color(value)).pack(side=tk.LEFT, padx=5)

# Save and Load Buttons
ttk.Button(color_frame, text="Save Settings", command=save_settings).pack(side=tk.LEFT, padx=5)
ttk.Button(color_frame, text="Load Settings", command=load_settings).pack(side=tk.LEFT, padx=5)

# Target Color Label
color_label = ttk.Label(color_frame, text="Target Color: Red")
color_label.pack(side=tk.LEFT, padx=5)

# HSV Adjustment Frame
hsv_frame = ttk.Frame(root)
hsv_frame.pack(pady=5)

lower_h = tk.IntVar(value=0)
upper_h = tk.IntVar(value=5)
lower_s = tk.IntVar(value=200)
upper_s = tk.IntVar(value=255)
lower_v = tk.IntVar(value=180)
upper_v = tk.IntVar(value=255)

# Labels to display current slider values
lower_h_label_var = tk.StringVar(value=str(lower_h.get()))
upper_h_label_var = tk.StringVar(value=str(upper_h.get()))
lower_s_label_var = tk.StringVar(value=str(lower_s.get()))
upper_s_label_var = tk.StringVar(value=str(upper_s.get()))
lower_v_label_var = tk.StringVar(value=str(lower_v.get()))
upper_v_label_var = tk.StringVar(value=str(upper_v.get()))

# Update labels when slider values change
lower_h.trace_add("write", lambda *args: lower_h_label_var.set(str(lower_h.get())))
upper_h.trace_add("write", lambda *args: upper_h_label_var.set(str(upper_h.get())))
lower_s.trace_add("write", lambda *args: lower_s_label_var.set(str(lower_s.get())))
upper_s.trace_add("write", lambda *args: upper_s_label_var.set(str(upper_s.get())))
lower_v.trace_add("write", lambda *args: lower_v_label_var.set(str(lower_v.get())))
upper_v.trace_add("write", lambda *args: upper_v_label_var.set(str(upper_v.get())))

# Lower H
ttk.Label(hsv_frame, text="Lower H:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=lower_h).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=lower_h_label_var).pack(side=tk.LEFT, padx=5)

# Upper H
ttk.Label(hsv_frame, text="Upper H:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=upper_h).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=upper_h_label_var).pack(side=tk.LEFT, padx=5)

# Lower S
ttk.Label(hsv_frame, text="Lower S:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=lower_s).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=lower_s_label_var).pack(side=tk.LEFT, padx=5)

# Upper S
ttk.Label(hsv_frame, text="Upper S:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=upper_s).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=upper_s_label_var).pack(side=tk.LEFT, padx=5)

# Lower V
ttk.Label(hsv_frame, text="Lower V:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=lower_v).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=lower_v_label_var).pack(side=tk.LEFT, padx=5)

# Upper V
ttk.Label(hsv_frame, text="Upper V:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=upper_v).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, textvariable=upper_v_label_var).pack(side=tk.LEFT, padx=5)

# Status
status_label = ttk.Label(root, text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
status_label.pack(pady=5)

# Terminal window
terminal_frame = ttk.Frame(root)
terminal_frame.pack(pady=10, fill=tk.BOTH, expand=True)

terminal_text = tk.Text(terminal_frame, height=10, width=80, bg="black", fg="white", font=("Courier", 10))
terminal_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = ttk.Scrollbar(terminal_frame, orient="vertical", command=terminal_text.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
terminal_text.config(yscrollcommand=scrollbar.set)

class RedirectText:
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert(tk.END, text)
        self.widget.see(tk.END)

    def flush(self):
        pass

sys.stdout = RedirectText(terminal_text)

# Screen capture function
def update_screen():
    global screen_frame
    with mss() as sct:
        monitor = {"top": capture_top, "left": capture_left, "width": capture_width, "height": capture_height}
        while not stop_event.is_set():
            if not frame_queue.full():
                screen = sct.grab(monitor)
                frame = np.array(screen)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                with screen_lock:
                    screen_frame = frame
                frame_queue.put(frame)
            time.sleep(0.002)

# Core aim assist processing function
def capture_and_process():
    global assist_enabled, curr_x, curr_y, screen_frame, last_detected_time, target_x, target_y, last_move_time, mouse_speed, assist_delay
    while not stop_event.is_set():
        with screen_lock:
            frame = screen_frame
        if frame is None or not assist_enabled:
            time.sleep(0.002)
            continue
        
        # Use the entire frame as the region since it's already 400x225
        region = frame
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Color detection for enemy
        lower_bound = np.array([lower_h.get(), lower_s.get(), lower_v.get()])
        upper_bound = np.array([upper_h.get(), upper_s.get(), upper_v.get()])
        enemy_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Color detection for props (tune these values based on your game)
        lower_prop = np.array([25, 150, 150])  # Example range, adjust as needed
        upper_prop = np.array([35, 255, 255])  # Example range, adjust as needed
        prop_mask = cv2.inRange(hsv, lower_prop, upper_prop)
        
        # Final mask: enemy mask minus prop mask
        final_mask = cv2.bitwise_and(enemy_mask, cv2.bitwise_not(prop_mask))
        
        # Apply Canny edge detection for better outline detection
        edges = cv2.Canny(final_mask, 100, 200)
        
        # Find target points from edges, fallback to mask
        points = cv2.findNonZero(edges)
        if points is None:
            points = cv2.findNonZero(final_mask)  # Use mask if no edges found
        
        target_x, target_y = None, None
        
        if points is not None:
            points = points.reshape(-1, 2)
            points = sorted(points, key=lambda p: p[1])  # Sort by y-coordinate
            if points:
                min_y = points[0][1]
                top_points = [p for p in points if p[1] == min_y]
                avg_x = sum(p[0] for p in top_points) / len(top_points)
                target_x = min(1920, max(0, capture_left + avg_x))
                target_y = min(1080, max(0, capture_top + min_y + 5))  # Offset 5 pixels down
        
        if target_x is not None and target_y is not None:
            # Calculate distance from screen center (960, 540)
            dx = target_x - 960
            dy = target_y - 540
            distance = math.sqrt(dx**2 + dy**2)
            
            # Only move if target is within assist_range
            if distance <= assist_range:
                dx_move = target_x - curr_x
                dy_move = target_y - curr_y
                move_dist = (dx_move**2 + dy_move**2)**0.5
                
                if move_dist > 0:  # Only move if the distance is significant
                    current_time = time.time()
                    delta_time = current_time - last_move_time
                    last_move_time = current_time
                    
                    if delta_time <= 0:
                        delta_time = 0.001
                    
                    print(f"Target: ({target_x}, {target_y}), Distance from center: {distance:.1f}, Move distance: {move_dist:.1f}")
                    print(f"delta_time: {delta_time:.3f}, mouse_speed: {mouse_speed}, assist_delay: {assist_delay}")
                    
                    if current_time - last_detected_time >= (assist_delay / 1000.0):
                        base_speed = mouse_speed * 10
                        tension_factor = mouse_speed / 250.0
                        step = tension_factor * move_dist
                        step = max(5, min(step, move_dist))
                        step *= delta_time * base_speed / 10.0
                        if step > move_dist:
                            step = move_dist
                        
                        print(f"Step: {step:.1f}")
                        
                        move_x = curr_x + (dx_move / move_dist) * step
                        move_y = curr_y + (dy_move / move_dist) * step
                        new_x, new_y = max(0, min(int(move_x), 1920)), max(0, min(int(move_y), 1080))
                        vnc.mouseMove(new_x, new_y)
                        curr_x, curr_y = new_x, new_y
                        last_detected_time = current_time
            else:
                print(f"Target outside range: {distance:.1f} > {assist_range}")
        else:
            last_move_time = time.time()
            
        # Update mask display with final_mask
        if show_mask and not mask_queue.full():
            mask_queue.put(cv2.resize(final_mask, (320, 180)))

        # Toggle assist off with Ctrl+Shift+D
        if keyboard.is_pressed("ctrl+shift+d"):
            assist_enabled = False
            assist_var.set(False)
            status_label.config(text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
            time.sleep(0.2)

# Mask display function
def display_mask():
    while not stop_event.is_set():
        if not mask_queue.empty():
            mask = mask_queue.get()
            cv2.imshow("Mask", mask)
            cv2.waitKey(1)
        time.sleep(0.033)

# GUI update function
def update_gui():
    global curr_x, curr_y, assist_range, target_x, target_y, assist_enabled
    if not stop_event.is_set():
        try:
            if not frame_queue.empty():
                frame = frame_queue.get_nowait()
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Resize the frame to fit the GUI (e.g., 640x360)
                    resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    canvas_width, canvas_height = 640, 360
                    
                    # Define the center of the capture region in GUI coordinates
                    center_x_scaled = int(200 * (canvas_width / capture_width))  # capture_width is the width of the capture region
                    center_y_scaled = int(112 * (canvas_height / capture_height))  # capture_height is the height of the capture region
                    
                    # Always draw the range circle (green), scaled to GUI size
                    range_scaled = int(assist_range * (canvas_width / capture_width))
                    cv2.circle(resized, (center_x_scaled, center_y_scaled), range_scaled, (0, 255, 0), 2)
                    
                    # Draw the tracer line (red) and target circle (blue) only if assist is enabled and a target is detected
                    if assist_enabled and target_x is not None and target_y is not None:
                        # Convert full-screen target coordinates to captured frame coordinates
                        target_x_cap = target_x - capture_left  # capture_left is the x-offset of the capture region
                        target_y_cap = target_y - capture_top   # capture_top is the y-offset of the capture region
                        # Scale to GUI coordinates
                        target_x_scaled = int(target_x_cap * (canvas_width / capture_width))
                        target_y_scaled = int(target_y_cap * (canvas_height / capture_height))
                        # Clamp coordinates to stay within GUI bounds
                        target_x_scaled = max(0, min(target_x_scaled, canvas_width - 1))
                        target_y_scaled = max(0, min(target_y_scaled, canvas_height - 1))
                        # Draw the tracer line (red) and target circle (blue)
                        cv2.line(resized, (center_x_scaled, center_y_scaled), (target_x_scaled, target_y_scaled), (0, 0, 255), 2)
                        cv2.circle(resized, (target_x_scaled, target_y_scaled), 5, (255, 0, 0), -1)
                    
                    # Convert the frame to a Tkinter-compatible image
                    success, encoded = cv2.imencode('.ppm', resized)
                    if success:
                        img = tk.PhotoImage(data=encoded.tobytes())
                        canvas.create_image(0, 0, image=img, anchor='nw')
                        canvas.image = img
                    else:
                        print("Failed to encode image to PPM")
                else:
                    print("Invalid image format:", frame.shape)
        except Exception as e:
            print(f"GUI update error: {e}")
        root.after(33, update_gui)  # Update every ~33ms (30 FPS)

# Thread management and cleanup
screen_thread = threading.Thread(target=update_screen)
process_thread = threading.Thread(target=capture_and_process)
mask_thread = threading.Thread(target=display_mask)

# Start threads
screen_thread.start()
process_thread.start()
if show_mask:
    mask_thread.start()

def on_closing():
    stop_event.set()
    screen_thread.join()
    process_thread.join()
    if show_mask:
        mask_thread.join()
    try:
        vnc.disconnect()
    except:
        pass
    cv2.destroyAllWindows()
    print("Application closed.")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
print("Starting GUI updates.")
update_gui()
root.mainloop()