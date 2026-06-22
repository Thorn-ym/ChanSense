import sys
import os
import time
import re
import serial
import serial.tools.list_ports

def get_serial_ports():
    return serial.tools.list_ports.comports()

def main():
    print("====================================================")
    print("         WiFi CSI Frame Rate (FPS) Tester           ")
    print("====================================================")
    
    ports = get_serial_ports()
    if not ports:
        print("No serial ports found! Please connect your ESP32-C6 (or CSI transmitting board).")
        sys.exit(1)
        
    print("Available Serial Ports:")
    for idx, port in enumerate(ports):
        print(f"[{idx}] {port.device} - {port.description}")
        
    while True:
        try:
            choice = input(f"\nSelect port [0-{len(ports)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(ports):
                port_name = ports[idx].device
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Please enter a valid number.")
            
    # Baud rates choices
    print("\nBaud Rate Options:")
    print("[0] 115200 (Default)")
    print("[1] 921600 (High Speed)")
    print("[2] Custom")
    baud_choice = input("Select baud rate [0-2]: ").strip()
    if baud_choice == '1':
        baud_rate = 921600
    elif baud_choice == '2':
        try:
            baud_rate = int(input("Enter custom baud rate: ").strip())
        except ValueError:
            baud_rate = 115200
    else:
        baud_rate = 115200
        
    print(f"\nOpening {port_name} at {baud_rate} baud...")
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=0.5)
        print("Port successfully opened! Listening for CSI frames...")
    except Exception as e:
        print(f"Error opening port: {e}")
        sys.exit(1)
        
    # Matching regex patterns identical to get_tool_fusion.py
    pattern = re.compile(r'data:\s*\[([^\]]+)\]')
    fallback_pattern = re.compile(r'\[([^\]]+)\]')
    
    print("\nPress Ctrl+C to stop the test.\n")
    print("-" * 55)
    print(f"{'Time':<12} | {'FPS (Hz)':<10} | {'Total Frames':<12} | {'Avg Subcarriers/Frame'}")
    print("-" * 55)
    
    total_frames = 0
    fps_counter = 0
    subcarrier_counts = []
    
    start_time = time.time()
    last_report_time = start_time
    
    try:
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                        
                    match = pattern.search(line) or fallback_pattern.search(line)
                    if match:
                        data_str = match.group(1)
                        # Count subcarriers (interleaved, so divide by 2 for complex count)
                        data_list = [x.strip() for x in data_str.split(',') if x.strip()]
                        subcarriers_len = len(data_list) // 2
                        
                        fps_counter += 1
                        total_frames += 1
                        subcarrier_counts.append(subcarriers_len)
                except Exception as e:
                    # Ignore decode or read errors during high speed stream
                    pass
            
            # Report every 1.0 second
            current_time = time.time()
            if current_time - last_report_time >= 1.0:
                elapsed = current_time - last_report_time
                fps = fps_counter / elapsed
                
                avg_subcarriers = 0
                if subcarrier_counts:
                    avg_subcarriers = sum(subcarrier_counts) / len(subcarrier_counts)
                    
                local_time_str = time.strftime("%H:%M:%S", time.localtime(current_time))
                print(f"{local_time_str:<12} | {fps:<10.1f} | {total_frames:<12} | {avg_subcarriers:.1f}")
                
                # Reset interval stats
                fps_counter = 0
                subcarrier_counts = []
                last_report_time = current_time
                
            time.sleep(0.001)  # small sleep to prevent high CPU load
            
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        
    duration = time.time() - start_time
    print("-" * 55)
    print(f"Test Duration:   {duration:.1f} seconds")
    print(f"Total Received:  {total_frames} frames")
    if duration > 0:
        print(f"Average FPS:     {total_frames / duration:.1f} Hz")
        
    ser.close()

if __name__ == '__main__':
    main()
