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
import colorsys  # For RGB to HSV conversion
from queue import Queue

# Function to convert RGB to HSV
def rgb_to_hsv(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    return (int(h * 180), int(s * 255), int(v * 255))

# Function to update HSV based on RGB input
def update_color():
    try:
        r = int(r_entry.get())
        g = int(g_entry.get())
        b = int(b_entry.get())
        if 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255:
            hsv = rgb_to_hsv(r, g, b)
            lower_h.set(max(0, hsv[0] - 10))
            upper_h.set(min(180, hsv[0] + 10))
            lower_s.set(max(0, hsv[1] - 50))
            upper_s.set(min(255, hsv[1] + 50))
            lower_v.set(max(0, hsv[2] - 50))
            upper_v.set(min(255, hsv[2] + 50))
            color_label.config(text=f"Target Color: Custom (RGB: {r}, {g}, {b})")
            print(f"Updated target color to RGB: ({r}, {g}, {b})")
        else:
            raise ValueError("RGB values must be between 0 and 255.")
    except ValueError as e:
        print(f"Error: {e}. Please enter valid integers between 0 and 255.")

# Global variables
curr_x, curr_y = 960, 540  # Center of 1920x1080 screen
running = True
assist_enabled = False
assist_range = 1000  # Temporarily increased to test mouse movement (was 75)
mouse_speed = 19     # Match your image setting
assist_delay = 150   # Adjustable delay before assist kicks in (ms)
last_detected_time = 0  # Track when target enters range
last_send_time = 0   # Track last mouse movement time (for reference)
screen_frame = None
screen_lock = threading.Lock()  # Lock for synchronizing screen_frame access
vnc_connected = False  # Track VNC connection status
target_x, target_y = None, None  # Track detected target position for drawing line
frame_queue = Queue(maxsize=1)  # Queue for passing frames to GUI thread
mask_queue = Queue(maxsize=1)   # Queue for passing masks to a separate display thread
show_mask = True  # Flag to control Mask window display

# Setup VNC to Main PC (loopback configuration)
try:
    vnc = api.connect("127.0.0.1", password="200210")  # Match TightVNC Server password
    vnc_connected = True
    print("VNC Server connected successfully to 127.0.0.1:5900")
except Exception as e:
    print(f"Failed to connect to VNC: {e}")
    print("Please ensure TightVNC Server is running and configured correctly.")
    assist_enabled = False

# Function definitions
def update_range(val):
    global assist_range
    assist_range = int(float(val))
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
    global assist_enabled, curr_x, curr_y
    assist_enabled = assist_var.get()
    status_label.config(text=f"Assist: {'Enabled' if assist_enabled else 'Disabled'} | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
    
    # Perform a subtle mouse wiggle when enabling assist to confirm functionality
    if assist_enabled and vnc_connected:
        print("Aim assist enabled - performing subtle mouse movement to confirm functionality.")
        try:
            for i in range(2):  # Two small movements for a wiggle effect
                vnc.mouseMove(curr_x + 10, curr_y)  # Move right
                time.sleep(0.05)  # Small delay for smoothness
                vnc.mouseMove(curr_x - 10, curr_y)  # Move left
                time.sleep(0.05)
            vnc.mouseMove(curr_x, curr_y)  # Return to original position
            print("Mouse wiggle completed.")
        except Exception as e:
            print(f"Error during mouse wiggle: {e}")
            assist_enabled = False
            assist_var.set(False)
            status_label.config(text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")

# GUI Setup
root = tk.Tk()
root.title("Helper Assist Beta v1 (Main PC) - Python 3.10.11")
root.geometry("800x700")
print("Tkinter root initialized.")  # Debug to confirm single initialization

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
range_slider = ttk.Scale(control_frame, from_=20, to=300, orient=tk.HORIZONTAL, command=update_range)
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

ttk.Label(color_frame, text="Target Color (RGB):").pack(side=tk.LEFT, padx=5)
r_entry = tk.Entry(color_frame, width=5)
r_entry.insert(0, "255")  # Default to red (255, 0, 0)
r_entry.pack(side=tk.LEFT, padx=2)
g_entry = tk.Entry(color_frame, width=5)
g_entry.insert(0, "0")
g_entry.pack(side=tk.LEFT, padx=2)
b_entry = tk.Entry(color_frame, width=5)
b_entry.insert(0, "0")
b_entry.pack(side=tk.LEFT, padx=2)
ttk.Button(color_frame, text="Update Color", command=update_color).pack(side=tk.LEFT, padx=5)
color_label = ttk.Label(color_frame, text="Target Color: Custom (RGB: 255, 0, 0)")
color_label.pack(side=tk.LEFT, padx=5)

# HSV Adjustment Frame
hsv_frame = ttk.Frame(root)
hsv_frame.pack(pady=5)

# Default HSV range for red (tighter range for better detection)
lower_h = tk.IntVar(value=0)
upper_h = tk.IntVar(value=10)
lower_s = tk.IntVar(value=100)
upper_s = tk.IntVar(value=255)
lower_v = tk.IntVar(value=100)
upper_v = tk.IntVar(value=255)

ttk.Label(hsv_frame, text="Lower H:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=lower_h).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, text="Upper H:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=180, orient=tk.HORIZONTAL, variable=upper_h).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, text="Lower S:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=lower_s).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, text="Upper S:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=upper_s).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, text="Lower V:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=lower_v).pack(side=tk.LEFT, padx=5)
ttk.Label(hsv_frame, text="Upper V:").pack(side=tk.LEFT, padx=5)
ttk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=upper_v).pack(side=tk.LEFT, padx=5)

# Status
status_label = ttk.Label(root, text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
status_label.pack(pady=5)

def update_screen():
    global screen_frame, running
    with mss() as sct:
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        while running:
            if not frame_queue.full():
                screen = sct.grab(monitor)
                frame = np.array(screen)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to BGR
                with screen_lock:
                    screen_frame = frame
                frame_queue.put(frame)  # Pass frame to GUI
            time.sleep(0.033)  # ~30 FPS to reduce load

def capture_and_process():
    global assist_enabled, curr_x, curr_y, screen_frame, running, last_send_time, target_x, target_y, last_detected_time
    while running:
        with screen_lock:
            frame = screen_frame
        if frame is None or not assist_enabled:
            time.sleep(0.033)  # ~30 FPS to reduce load
            continue
        
        center_x, center_y = 960, 540
        region = frame[center_y-112:center_y+113, center_x-200:center_x+200]  # 400x225
        print(f"Processing region: center ({center_x}, {center_y}), size 400x225")
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([lower_h.get(), lower_s.get(), lower_v.get()])
        upper_bound = np.array([upper_h.get(), upper_s.get(), upper_v.get()])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 30:  # Lowered threshold to detect smaller targets
                x, y, w, h = cv2.boundingRect(largest)
                region_target_x = x + w // 2  # Relative to region (0-400)
                region_target_y = y + h // 2  # Relative to region (0-225)
                target_x = min(1920, max(0, (center_x - 200) + region_target_x))  # Correct offset
                target_y = min(1080, max(0, (center_y - 112) + region_target_y))  # Correct offset
                print(f"Target detected at region (x: {region_target_x}, y: {region_target_y}), "
                      f"screen (x: {target_x}, y: {target_y})")

                dx, dy = target_x - curr_x, target_y - curr_y
                dist = (dx**2 + dy**2)**0.5
                print(f"Distance to target: {dist}, Range: {assist_range}")

                if dist > 0 and dist <= assist_range:
                    current_time = time.time()
                    if current_time - last_detected_time >= (assist_delay / 1000.0):
                        # Dynamic movement with speed adjustment and dampening
                        movement_factor = min(1.0, 0.1 + (assist_range - dist) / assist_range)  # Lightens as it nears
                        speed_factor = mouse_speed / 100.0  # Apply mouse speed
                        # Dampen movement when very close to prevent sticking
                        if dist < assist_range * 0.1:  # Within 10% of range, reduce force
                            movement_factor *= (dist / (assist_range * 0.1))  # Exponential decay
                        assist_x = int(dx * movement_factor * speed_factor)
                        assist_y = int(dy * movement_factor * speed_factor)

                        new_x = curr_x + assist_x
                        new_y = curr_y + assist_y
                        new_x, new_y = max(0, min(new_x, 1920)), max(0, min(new_y, 1080))

                        try:
                            vnc.mouseMove(new_x, new_y)
                            print(f"Mouse moved to (x: {new_x}, y: {new_y}) from (x: {curr_x}, y: {curr_y})")
                            curr_x, curr_y = new_x, new_y
                        except Exception as e:
                            print(f"Error moving mouse: {e}")
                            assist_enabled = False
                            assist_var.set(False)
                            status_label.config(text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
                    else:
                        print(f"Waiting for reaction delay: {current_time - last_detected_time:.3f}s / {assist_delay / 1000.0}s")
                else:
                    last_detected_time = time.time()  # Reset timer when target is out of range
                    print("Target out of range for mouse movement.")
            else:
                target_x, target_y = None, None
                print("Target too small.")
                last_detected_time = time.time()  # Reset timer on small target
        else:
            target_x, target_y = None, None
            print("No target detected.")
            last_detected_time = time.time()  # Reset timer on no target

        if keyboard.is_pressed("ctrl+shift+d"):
            assist_enabled = False
            assist_var.set(False)
            status_label.config(text=f"Assist: Disabled | VNC: {'Connected' if vnc_connected else 'Disconnected'}")
            target_x, target_y = None, None  # Clear target on disable
            time.sleep(0.2)  # Debounce

        # Pass mask to a separate queue for display
        if show_mask and not mask_queue.full():
            mask_queue.put(cv2.resize(mask, (320, 180)))

def display_mask():
    while running:
        if not mask_queue.empty():
            mask = mask_queue.get()
            cv2.imshow("Mask", mask)
            cv2.waitKey(1)  # Non-blocking
        time.sleep(0.033)  # ~30 FPS to reduce load

def update_gui():
    global curr_x, curr_y, assist_range, target_x, target_y
    if running:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get_nowait()
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    resized = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
                    canvas_width, canvas_height = 640, 360  # Explicitly define canvas dimensions
                    center_x_scaled = int(960 * canvas_width / 1920)  # Fixed center x
                    center_y_scaled = int(540 * canvas_height / 1080)  # Fixed center y
                    range_scaled = int(assist_range * canvas_width / 1920)
                    if assist_enabled:
                        cv2.circle(resized, (center_x_scaled, center_y_scaled),
                                  range_scaled, (0, 255, 0), 2)  # Thicker green circle fixed at center
                        # Draw line regardless of range
                        if target_x is not None and target_y is not None:
                            target_x_scaled = int(target_x * canvas_width / 1920)
                            target_y_scaled = int(target_y * canvas_height / 1080)
                            target_x_scaled = max(0, min(target_x_scaled, canvas_width - 1))
                            target_y_scaled = max(0, min(target_y_scaled, canvas_height - 1))
                            print(f"Drawing line from (curr_x: {center_x_scaled}, curr_y: {center_y_scaled}) "
                                  f"to (target_x: {target_x_scaled}, target_y: {target_y_scaled})")
                            # Draw a thicker red line for visibility
                            cv2.line(resized, (center_x_scaled, center_y_scaled),
                                     (target_x_scaled, target_y_scaled), (0, 0, 255), 2)
                            # Fallback: Draw a green line to ensure visibility
                            cv2.line(resized, (center_x_scaled, center_y_scaled),
                                     (target_x_scaled, target_y_scaled), (0, 255, 0), 1)
                    success, encoded = cv2.imencode('.ppm', resized)
                    if success:
                        img = tk.PhotoImage(data=encoded.tobytes())
                        canvas.create_image(0, 0, image=img, anchor='nw')
                        canvas.image = img  # Prevent garbage collection
                    else:
                        print("Failed to encode image to PPM")
                else:
                    print("Invalid image format:", frame.shape)
        except Exception as e:
            print(f"GUI update error: {e}")
        root.after(33, update_gui)  # ~30 FPS to reduce load

# Start threads
threading.Thread(target=update_screen, daemon=True).start()
threading.Thread(target=capture_and_process, daemon=True).start()
if show_mask:
    threading.Thread(target=display_mask, daemon=True).start()

def on_closing():
    global running
    running = False
    time.sleep(0.2)  # Give threads a moment to stop
    try:
        vnc.disconnect()
    except:
        pass
    cv2.destroyAllWindows()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start GUI updates in the main thread
print("Starting GUI updates.")
update_gui()

root.mainloop()