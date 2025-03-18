import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QSlider, QCheckBox, QPushButton, QComboBox, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap
import threading
import keyboard
import time
import random
import colorsys
import math
import json
import sys
import socket
import NDIlib as ndi
import numba
import torch
from queue import Queue
import os

# Global variables
curr_x, curr_y = 960, 540
assist_enabled = False
assist_range = 30  # Default 30px in GUI
mouse_speed = 1
assist_delay = 150
last_detected_time = 0
last_send_time = 0
screen_frame = None
screen_lock = threading.Lock()
target_x, target_y = None, None
new_x, new_y = 960, 540  # For GUI overlay, reset to center
frame_queue = Queue(maxsize=5)
mask_queue = Queue(maxsize=5)
mask_image_queue = Queue(maxsize=5)
show_mask = True
stop_event = threading.Event()
last_target_x, last_target_y = 960, 540  # Initialize to center
last_move_time = time.time()

# Screen resolution (adjust to your screen size)
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Capture region parameters (100x100, centered at 960, 540)
capture_left = 910  # 960 - 50
capture_top = 490   # 540 - 50
capture_width = 100
capture_height = 100

# Get the directory where the script or executable is running
if getattr(sys, 'frozen', False):
    base_path = os.path.dirname(sys.executable)  # For when it’s an executable
else:
    base_path = os.path.dirname(__file__)        # For when running the script directly

# Build the path to best.pt
model_path = os.path.join(base_path, 'best.pt')

# UDP setup for mouse control
UDP_IP = input('Enter Main PC IP (type ipconfig on Main PC): ')
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print(f"UDP setup for mouse control to {UDP_IP}:{UDP_PORT}")

# Load YOLOv5 model (nano version for low-end PCs)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Numba-accelerated color detection
@numba.jit(nopython=True)
def color_distance(hsv_frame, lower_bound, upper_bound, height, width):
    target_x, target_y = -1, -1
    min_dist = float('inf')
    for y in range(height):
        for x in range(width):
            h, s, v = hsv_frame[y, x]
            if (lower_bound[0] <= h <= upper_bound[0] and
                lower_bound[1] <= s <= upper_bound[1] and
                lower_bound[2] <= v <= upper_bound[2]):
                dist = math.sqrt((h - (lower_bound[0] + upper_bound[0]) / 2)**2 +
                                 (s - (lower_bound[1] + upper_bound[1]) / 2)**2 +
                                 (v - (lower_bound[2] + upper_bound[2]) / 2)**2)
                if dist < min_dist:
                    min_dist = dist
                    target_x, target_y = x, y
    return target_x, target_y, min_dist

# Color preset functions
def set_pre_tuned_color(color):
    if color == "Red":
        lower_h.setValue(151)
        upper_h.setValue(179)
        lower_s.setValue(163)
        upper_s.setValue(255)
        lower_v.setValue(58)
        upper_v.setValue(222)
    elif color == "Purple":
        lower_h.setValue(151)
        upper_h.setValue(179)
        lower_s.setValue(163)
        upper_s.setValue(255)
        lower_v.setValue(58)
        upper_v.setValue(222)
    elif color == "Yellow":
        lower_h.setValue(20)
        upper_h.setValue(40)
        lower_s.setValue(100)
        upper_s.setValue(255)
        lower_v.setValue(100)
        upper_v.setValue(255)
    update_color_label(color)
    print(f"Set pre-tuned color: {color}")

def update_color_label(color):
    window.color_combo.setToolTip(f"Target Color: {color}")

# NDI Capture Thread
class NDIThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def run(self):
        global screen_frame
        try:
            if not ndi.initialize():
                print("Failed to initialize NDI.")
                return
            finder = ndi.find_create_v2()
            if not finder:
                print("Failed to create NDI finder.")
                ndi.destroy()
                return
            selected_source = None
            attempt = 0
            while not stop_event.is_set() and not selected_source and attempt < 60:  # Retry ~2min
                sources = ndi.find_get_current_sources(finder)
                print(f"Attempt {attempt + 1}: Found sources: {[s.ndi_name for s in sources]}")
                for source in sources:
                    if 'OBS' in source.ndi_name:
                        selected_source = source
                        print(f"Selected NDI source: {source.ndi_name}")
                        break
                if not selected_source:
                    print("No OBS NDI source found—retrying...")
                    time.sleep(1)
                    attempt += 1
            if not selected_source:
                print("Failed to find OBS NDI source after retries.")
                return
            receiver = ndi.recv_create_v3()
            ndi.recv_connect(receiver, selected_source)
            while not stop_event.is_set():
                frame_type, video_frame, _, _ = ndi.recv_capture_v2(receiver, 5000)
                if frame_type == ndi.FRAME_TYPE_VIDEO:
                    frame_data = np.frombuffer(video_frame.data, dtype=np.uint8)
                    frame = frame_data.reshape((video_frame.yres, video_frame.line_stride_in_bytes // 4, 4))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    # Crop to 100x100 centered region (assuming 1920x1080 input)
                    h, w = frame.shape[:2]
                    center_x, center_y = w // 2, h // 2
                    cropped_frame = frame[
                        max(0, center_y - 50):min(h, center_y + 50),
                        max(0, center_x - 50):min(w, center_x + 50)
                    ]
                    with screen_lock:
                        screen_frame = cropped_frame
                    if not frame_queue.full():
                        frame_queue.put(cropped_frame)
                        self.frame_signal.emit(cropped_frame)
                    else:
                        print("Frame queue full—skipping frame")
                else:
                    print("No video frame received from NDI")
                time.sleep(0.001)
            ndi.recv_destroy(receiver)
            ndi.find_destroy(finder)
            ndi.destroy()
        except Exception as e:
            print(f"NDI error: {e}")

# Aim Assist Thread
class AimThread(QThread):
    def run(self):
        global assist_enabled, screen_frame, last_detected_time, target_x, target_y, last_move_time, mouse_speed, assist_delay, SCREEN_WIDTH, SCREEN_HEIGHT, capture_width, capture_height
        while not stop_event.is_set():
            with screen_lock:
                frame = screen_frame
            if frame is None or not assist_enabled:
                print("No frame or assist disabled—waiting...")
                time.sleep(0.001) # Reduce Delay for more frequent updates
                continue
            
            print("Processing frame...")
            cropped_frame = frame  # Already 100x100 from NDIThread
            cv2.imwrite("test_crop.jpg", cropped_frame)  # Save for debugging
            
            # Step 1: YOLOv5 Detection (Primary)
            target_x, target_y = None, None
            try:
                results = model(cropped_frame)
                detections = results.xyxy[0].cpu().numpy()
                print("YOLOv5 detections:", detections)
                if len(detections) > 0:
                    best = detections[detections[:, 4].argmax()]
                    x_min, y_min, x_max, y_max, conf, _ = best
                    target_x = int((x_min + x_max) / 2)
                    target_y = int((y_min + y_max) / 2)
                    print(f"YOLOv5 Target: ({target_x}, {target_y}), Confidence: {conf}")
            except Exception as e:
                print(f"YOLOv5 error: {e}")
            
            # Step 2: Color Detection (Backup)
            if target_x is None:  # YOLOv5 failed, try Numba
                hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
                lower_bound = np.array([lower_h.value(), lower_s.value(), lower_v.value()])
                upper_bound = np.array([upper_h.value(), upper_s.value(), upper_v.value()])
                tx, ty, dist = color_distance(hsv, lower_bound, upper_bound, 100, 100)
                print(f"Numba detection: tx={tx}, ty={ty}, dist={dist}")
                if tx != -1 and ty != -1 and dist < 50:  # Valid detection within range
                    target_x, target_y = tx, ty
                    print("Using Numba target")
                else:
                    print("No target detected by Numba")
            else:
                print("Using YOLOv5 target")
            
            # Step 3: Calculate Mouse Nudge
            dx, dy = 0, 0
            if target_x is not None:
                center_x, center_y = 50, 50  # Center of 100x100 frame
                dx_frame = target_x - center_x
                dy_frame = target_y - center_y
                distance_frame = math.sqrt(dx_frame**2 + dy_frame**2)
                if distance_frame <= assist_range:
                    # Scale offset to screen resolution
                    scale_x = SCREEN_WIDTH / capture_width  # e.g., 1920 / 100 = 19.2
                    scale_y = SCREEN_HEIGHT / capture_height  # e.g., 1080 / 100 = 10.8
                    dx_screen = dx_frame * scale_x
                    dy_screen = dy_frame * scale_y
                    # Adjust for sensitivity
                    sensitivity_multiplier = 0.5  # Tune this value (e.g., 0.05 to 0.5)
                    dx_nudge = dx_screen * sensitivity_multiplier
                    dy_nudge = dy_screen * sensitivity_multiplier
                    # Cap the nudge to prevent extreme movements
                    max_nudge = 1  # Pixels
                    dx_nudge = max(-max_nudge, min(dx_nudge, max_nudge))
                    dy_nudge = max(-max_nudge, min(dy_nudge, max_nudge))
                    dx, dy = dx_nudge, dy_nudge
                    print(f"Nudge: dx={dx}, dy={dy}")
            
            # Step 4: Send UDP Packet
            try:
                data = json.dumps({"dx": dx, "dy": dy})
                sock.sendto(data.encode(), (UDP_IP, UDP_PORT))
                print(f"Sent UDP: dx={dx}, dy={dy}")
            except Exception as e:
                print(f"UDP error: {e}")
            
            time.sleep(0.001)  # Reduced CPU load

# GUI Class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magic Assist Beta v1 - Script PC")
        self.setGeometry(100, 100, 700, 700)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        video_mask_layout = QHBoxLayout()
        layout.addLayout(video_mask_layout)

        self.video_label = QLabel()
        self.video_label.setFixedSize(100, 100)
        video_mask_layout.addWidget(self.video_label)

        self.mask_label = QLabel()
        self.mask_label.setFixedSize(100, 100)
        video_mask_layout.addWidget(self.mask_label)

        self.assist_check = QCheckBox("Enable Aim Assist")
        self.assist_check.stateChanged.connect(self.toggle_assist)
        layout.addWidget(self.assist_check)

        self.range_slider = QSlider(Qt.Horizontal)
        self.range_slider.setMinimum(20)
        self.range_slider.setMaximum(100)
        self.range_slider.setValue(assist_range)
        self.range_slider.valueChanged.connect(self.update_range)
        layout.addWidget(QLabel("Range Radius (px)"))
        layout.addWidget(self.range_slider)
        self.range_value = QLabel(f"{assist_range}")
        layout.addWidget(self.range_value)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(mouse_speed)
        self.speed_slider.valueChanged.connect(self.update_speed)
        layout.addWidget(QLabel("Mouse Speed"))
        layout.addWidget(self.speed_slider)
        self.speed_value = QLabel(f"{mouse_speed}")
        layout.addWidget(self.speed_value)

        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setMinimum(0)
        self.delay_slider.setMaximum(500)
        self.delay_slider.setValue(assist_delay)
        self.delay_slider.valueChanged.connect(self.update_delay)
        layout.addWidget(QLabel("Reaction Delay (ms)"))
        layout.addWidget(self.delay_slider)
        self.delay_value = QLabel(f"{assist_delay}")
        layout.addWidget(self.delay_value)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Red", "Purple", "Yellow"])
        self.color_combo.currentTextChanged.connect(self.set_color)
        layout.addWidget(QLabel("Target Color"))
        layout.addWidget(self.color_combo)

        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton("Load Settings")
        self.load_button.clicked.connect(self.load_settings)
        layout.addWidget(self.load_button)

        global lower_h, upper_h, lower_s, upper_s, lower_v, upper_v
        lower_h = QSlider(Qt.Horizontal); lower_h.setRange(0, 180); lower_h.setValue(0)
        upper_h = QSlider(Qt.Horizontal); upper_h.setRange(0, 180); upper_h.setValue(5)
        lower_s = QSlider(Qt.Horizontal); lower_s.setRange(0, 255); lower_s.setValue(200)
        upper_s = QSlider(Qt.Horizontal); upper_s.setRange(0, 255); upper_s.setValue(255)
        lower_v = QSlider(Qt.Horizontal); lower_v.setRange(0, 255); lower_v.setValue(180)
        upper_v = QSlider(Qt.Horizontal); upper_v.setRange(0, 255); upper_v.setValue(255)
        layout.addWidget(QLabel("Lower H")); layout.addWidget(lower_h); self.lower_h_value = QLabel("0"); layout.addWidget(self.lower_h_value)
        layout.addWidget(QLabel("Upper H")); layout.addWidget(upper_h); self.upper_h_value = QLabel("5"); layout.addWidget(self.upper_h_value)
        layout.addWidget(QLabel("Lower S")); layout.addWidget(lower_s); self.lower_s_value = QLabel("200"); layout.addWidget(self.lower_s_value)
        layout.addWidget(QLabel("Upper S")); layout.addWidget(upper_s); self.upper_s_value = QLabel("255"); layout.addWidget(self.upper_s_value)
        layout.addWidget(QLabel("Lower V")); layout.addWidget(lower_v); self.lower_v_value = QLabel("180"); layout.addWidget(self.lower_v_value)
        layout.addWidget(QLabel("Upper V")); layout.addWidget(upper_v); self.upper_v_value = QLabel("255"); layout.addWidget(self.upper_v_value)
        
        lower_h.valueChanged.connect(lambda val: self.lower_h_value.setText(str(val)))
        upper_h.valueChanged.connect(lambda val: self.upper_h_value.setText(str(val)))
        lower_s.valueChanged.connect(lambda val: self.lower_s_value.setText(str(val)))
        upper_s.valueChanged.connect(lambda val: self.upper_s_value.setText(str(val)))
        lower_v.valueChanged.connect(lambda val: self.lower_v_value.setText(str(val)))
        upper_v.valueChanged.connect(lambda val: self.upper_v_value.setText(str(val)))

        self.ndi_thread = NDIThread()
        self.ndi_thread.frame_signal.connect(self.update_frame)
        self.ndi_thread.start()

        self.aim_thread = AimThread()
        self.aim_thread.start()

        self.mask_timer = QTimer()
        self.mask_timer.timeout.connect(self.update_mask)
        self.mask_timer.start(33)

    def update_frame(self, frame):
        global target_x, target_y, assist_range
        print("Updating video frame")
        while not frame_queue.empty():
            frame = frame_queue.get()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        center_x, center_y = 50, 50  # Center of 100x100 frame
        cv2.circle(frame_rgb, (center_x, center_y), 2, (0, 0, 255), -1)  # Red dot
        
        if target_x is not None and target_y is not None:
            tx, ty = int(target_x), int(target_y)
            cv2.line(frame_rgb, (center_x, center_y), (tx, ty), (0, 255, 0), 1)
            cv2.circle(frame_rgb, (tx, ty), 5, (0, 255, 0), -1)
        
        cv2.circle(frame_rgb, (center_x, center_y), assist_range, (0, 255, 0), 1)
        
        qimage = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], frame_rgb.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimage))

    def update_mask(self):
        while not mask_image_queue.empty():
            mask_rgb = mask_image_queue.get_nowait()
            print("Updating mask frame")
            qimage = QImage(mask_rgb.data, mask_rgb.shape[1], mask_rgb.shape[0], mask_rgb.strides[0], QImage.Format_RGB888)
            self.mask_label.setPixmap(QPixmap.fromImage(qimage))

    def toggle_assist(self, state):
        global assist_enabled
        assist_enabled = state == Qt.Checked
        self.assist_check.setText(f"Enable Aim Assist: {'On' if assist_enabled else 'Off'}")

    def update_range(self, val):
        global assist_range
        assist_range = val
        self.range_value.setText(str(val))

    def update_speed(self, val):
        global mouse_speed
        mouse_speed = val
        self.speed_value.setText(str(val))

    def update_delay(self, val):
        global assist_delay
        assist_delay = val
        self.delay_value.setText(str(val))

    def set_color(self, color):
        set_pre_tuned_color(color)

    def save_settings(self):
        settings = {
            "hsv": {
                "lower_h": lower_h.value(),
                "upper_h": upper_h.value(),
                "lower_s": lower_s.value(),
                "upper_s": upper_s.value(),
                "lower_v": lower_v.value(),
                "upper_v": upper_v.value()
            }
        }
        try:
            with open("settings.json", "w") as f:
                json.dump(settings, f, indent=4)
            print("Settings saved to settings.json")
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self):
        try:
            with open("settings.json", "r") as f:
                settings = json.load(f)
            lower_h.setValue(settings["hsv"]["lower_h"])
            upper_h.setValue(settings["hsv"]["upper_h"])
            lower_s.setValue(settings["hsv"]["lower_s"])
            upper_s.setValue(settings["hsv"]["upper_s"])
            lower_v.setValue(settings["hsv"]["lower_v"])
            upper_v.setValue(settings["hsv"]["upper_v"])
            update_color_label(self.color_combo.currentText())
            print("Settings loaded from settings.json")
        except Exception as e:
            print(f"Error loading settings: {e}. Using default values.")

    def closeEvent(self, event):
        stop_event.set()
        self.ndi_thread.wait()
        self.aim_thread.wait()
        event.accept()

# Main execution
app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()