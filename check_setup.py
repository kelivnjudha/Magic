import NDIlib as ndi
from vncdotool import api
import os
import time

def test_network(main_pc_ip):
    """Test if second PC can ping main PC."""
    print("\nTesting network connectivity...")
    response = os.system(f"ping -n 4 {main_pc_ip} >nul 2>&1")  # Windows ping, 4 attempts, quiet
    if response == 0:
        print(f"PASS: Successfully pinged {main_pc_ip}")
        return True
    else:
        print(f"FAIL: Could not ping {main_pc_ip}")
        print("  - Check: Both PCs on same network? (e.g., 192.168.68.x)")
        print("  - Check: Firewall allowing ICMP? (Try disabling firewall)")
        return False

def test_ndi(main_pc_ip):
    """Test if NDI 'OBS' source is detectable."""
    print("\nTesting NDI source detection...")
    try:
        if not ndi.initialize():
            print("FAIL: NDI library failed to initialize")
            print("  - Check: NDI Runtime installed on this PC?")
            return False

        finder = ndi.find_create_v2()
        if not finder:
            print("FAIL: Could not create NDI finder")
            ndi.destroy()
            return False

        sources = []
        for _ in range(15):  # Wait 15 seconds
            sources = ndi.find_get_current_sources(finder)
            if sources:
                break
            time.sleep(1)

        if not sources:
            print("FAIL: No NDI sources found after 15 seconds")
            print("  - Check: OBS running on main PC with NDI output 'OBS' enabled?")
            print("  - Check: Firewall allowing UDP 5960-5970?")
            ndi.find_destroy(finder)
            ndi.destroy()
            return False

        for source in sources:
            if "OBS" in source.ndi_name.lower():
                print(f"PASS: Found NDI source '{source.ndi_name}' at {source.address}")
                ndi.find_destroy(finder)
                ndi.destroy()
                return True

        print("FAIL: No 'OBS' NDI source found. Detected sources:", [s.ndi_name for s in sources])
        print("  - Check: OBS NDI output named 'OBS' in Tools → NDI Output Settings?")
        ndi.find_destroy(finder)
        ndi.destroy()
        return False

    except Exception as e:
        print(f"FAIL: NDI test error - {e}")
        return False

def test_vnc(main_pc_ip):
    """Test if VNC server is reachable."""
    print("\nTesting VNC connection...")
    try:
        client = api.connect(f"{main_pc_ip}:5900", timeout=5)
        print(f"PASS: Successfully connected to VNC server at {main_pc_ip}:5900")
        client.mouseMove(500, 500)  # Test move
        print("  - Mouse movement test sent successfully")
        client.disconnect()
        return True
    except Exception as e:
        print(f"FAIL: VNC connection failed - {e}")
        print("  - Check: TightVNC Server running on main PC?")
        print("  - Check: 'IP address to bind' set to 0.0.0.0 in TVNC config?")
        print("  - Check: Firewall allowing inbound TCP 5900 on main PC?")
        print("  - Check: Correct IP and port (5900)?")
        return False

def main():
    print("=== Dual-PC Setup Tester for main.exe ===")
    print("This script tests if your setup is ready to run the main aim assist script.")
    print("You’ll need: OBS NDI streaming 'OBS' and TightVNC Server on the main PC.")
    
    # Get main PC IP from user
    main_pc_ip = input("\nEnter the main PC’s IP address (e.g., 192.168.68.100): ").strip()
    
    # Run tests
    network_ok = test_network(main_pc_ip)
    ndi_ok = test_ndi(main_pc_ip)
    vnc_ok = test_vnc(main_pc_ip)

    # Summary
    print("\n=== Test Summary ===")
    if network_ok and ndi_ok and vnc_ok:
        print("SUCCESS: All tests passed! Your dual-PC setup is ready for main.exe.")
    else:
        print("FAILURE: One or more tests failed. Fix the issues above and retest.")
        if not network_ok:
            print("- Network test failed.")
        if not ndi_ok:
            print("- NDI test failed.")
        if not vnc_ok:
            print("- VNC test failed.")

if __name__ == "__main__":
    main()