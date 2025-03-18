import socket
import json
import time
from ctypes import windll, Structure, c_long, byref

# Define POINT structure for GetCursorPos
class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

UDP_IP = "0.0.0.0"
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.001)  # 100ms timeout
try:
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Mouse client bound to {UDP_IP}:{UDP_PORT}")
except Exception as e:
    print(f"Bind error: {e}")
    exit(1)

print("Mouse client running on Main PC...")
first_packet = True  # Flag for first packet

# Accumulators for fractional nudges
accum_dx = 0.0
accum_dy = 0.0

# Function to get current mouse position using ctypes
def get_mouse_position():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt.x, pt.y

while True:
    # print("Waiting for packet...")
    try:
        data, addr = sock.recvfrom(1024)
        packet_data = data.decode('utf-8')
        # print(f"Received packet from {addr}: {packet_data}")
        if first_packet:
            # print("Confirmed connection from magic.py—mouse ready!")
            first_packet = False
        
        # Parse the nudge vector
        nudge = json.loads(packet_data)
        dx, dy = nudge['dx'], nudge['dy']
        # print(f"Parsed nudge: dx={dx}, dy={dy}")
        
        # Accumulate the nudge
        accum_dx += dx
        accum_dy += dy
        # print(f"Accumulated: accum_dx={accum_dx}, accum_dy={accum_dy}")
        
        # Calculate integer movement
        move_x = int(accum_dx)
        move_y = int(accum_dy)
        
        if move_x != 0 or move_y != 0:
            # Get current mouse position
            curr_x, curr_y = get_mouse_position()
            # print(f"Current mouse position: ({curr_x}, {curr_y})")
            
            # Apply movement
            new_x = curr_x + move_x
            new_y = curr_y + move_y
            
            # Bound coords to screen (assuming 1920x1080 resolution)
            new_x = max(0, min(new_x, 1920 - 1))
            new_y = max(0, min(new_y, 1080 - 1))
            
            # print(f"Moving mouse to: ({new_x}, {new_y})")
            windll.user32.SetCursorPos(new_x, new_y)
            
            # Subtract the applied movement from accumulators
            accum_dx -= move_x
            accum_dy -= move_y
        else:
            # print("Nudge too small to move mouse")
            continue
        
    except socket.timeout:
        continue
        # print("No data received in 0.1s—waiting...")
    except Exception as e:
        print(f"Receive or move error: {e}")
        try:
            sock.close()
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(0.1)
            sock.bind((UDP_IP, UDP_PORT))
            print("Rebound socket")
        except Exception as re:
            print(f"Rebind error: {re}")