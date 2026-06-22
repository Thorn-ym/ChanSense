import os
import sys
import struct
import time
import re
import random
import numpy as np
import serial
import serial.tools.list_ports

# Add train_src to system path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_src"))
from dataset import load_dataset

# Mappings for display
LABEL_MAP = {0: 'cut (挥手)', 1: 'grip (抓握)', 2: 'draw_o (画圈)'}

def get_serial_ports():
    return serial.tools.list_ports.comports()

def main():
    print("====================================================")
    print("      ESP32-P4 1D-CNN Dataset Sample Sender         ")
    print("====================================================")
    
    # List available COM ports
    ports = get_serial_ports()
    if not ports:
        print("No serial ports found! Please connect your ESP32-P4 board.")
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
            
    baud_rate = 921600
    print(f"\nOpening {port_name} at {baud_rate} baud...")
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        print("Port successfully opened!")
    except Exception as e:
        print(f"Error opening port: {e}")
        sys.exit(1)
        
    # Load dataset
    transform_tool_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(transform_tool_dir, "dataset", "dataset_2026_6_10")
    print(f"\nLoading dataset from {dataset_dir}...")
    try:
        X_by_label = load_dataset(dataset_dir)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        ser.close()
        sys.exit(1)
        
    while True:
        print("\n====================================================")
        print("                  MAIN OPTIONS                      ")
        print("====================================================")
        print("[1] Send Single Sample (Manual Index)")
        print("[2] Run Batch Test Sweep (Continuous dataset evaluation)")
        print("[3] Exit")
        
        opt = input("Select an option [1-3]: ").strip()
        if opt == '1':
            print("\n-----------------------------")
            print("Classes available in dataset:")
            for k, v in LABEL_MAP.items():
                count = len(X_by_label[k]) if k in X_by_label else 0
                print(f"  [{k}] {v} - Count: {count} samples")
            
            try:
                class_choice = input("Select a class [0-2]: ").strip()
                class_idx = int(class_choice)
                if class_idx not in X_by_label:
                    print("Invalid class index.")
                    continue
                    
                samples = X_by_label[class_idx]
                max_idx = len(samples) - 1
                
                sample_choice = input(f"Select sample index [0-{max_idx}] (default 0): ").strip()
                if not sample_choice:
                    sample_idx = 0
                else:
                    sample_idx = int(sample_choice)
                    if not (0 <= sample_idx <= max_idx):
                        print(f"Index out of range! Must be between 0 and {max_idx}.")
                        continue
                        
                print(f"\nProcessing sample {sample_idx} from class '{LABEL_MAP[class_idx]}'")
                
                # Select 1 complex sample: shape (50, 114)
                complex_data = samples[sample_idx]  # Shape: (50, 114) complex
                
                # Serialize raw complex data (interleave imaginary and real parts)
                raw_floats = np.empty((50, 228), dtype=np.float32)
                raw_floats[:, 0::2] = complex_data.imag
                raw_floats[:, 1::2] = complex_data.real
                flat_features = raw_floats.flatten()
                
                # Construct framing payload
                header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
                data_bytes = struct.pack('<11400f', *flat_features)
                length_bytes = struct.pack('<I', len(data_bytes))
                checksum = sum(data_bytes) & 0xFFFFFFFF
                checksum_bytes = struct.pack('<I', checksum)
                packet = header + length_bytes + data_bytes + checksum_bytes
                
                print(f"Sending packet: Header=4B, Length={len(data_bytes)}B, Checksum=0x{checksum:08X} (Total: {len(packet)}B)")
                
                # Flush any stale log data
                ser.reset_input_buffer()
                
                # Write to serial port
                ser.write(packet)
                print("Packet sent successfully! Waiting for ESP32-P4 inference logs...\n")
                
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    if ser.in_waiting > 0:
                        try:
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            if line:
                                print(f"[Board Log] {line}")
                                if "Inference Result:" in line or "Inference failed" in line:
                                    break
                        except Exception as read_err:
                            print(f"Error reading logs: {read_err}")
                            break
                    time.sleep(0.01)
                
            except ValueError:
                print("Please enter valid numbers.")
            except Exception as e:
                print(f"An error occurred: {e}")
                
        elif opt == '2':
            print("\n----------------------------------------------------")
            print("Batch Test Sweep Settings:")
            print("Select class mode:")
            print("  [0] Sweep 'cut (挥手)' only")
            print("  [1] Sweep 'grip (抓握)' only")
            print("  [2] Sweep 'draw_o (画圈)' only")
            print("  [3] Sweep mixed samples (interleaved across all classes)")
            
            try:
                mode_choice = input("Select mode [0-3]: ").strip()
                if not mode_choice:
                    continue
                mode_choice = int(mode_choice)
                if mode_choice not in [0, 1, 2, 3]:
                    print("Invalid choice.")
                    continue
                
                count_input = input("How many samples to test? (default 20): ").strip()
                count = int(count_input) if count_input else 20
                
                delay_input = input("Delay between samples in seconds? (default 0.5): ").strip()
                delay = float(delay_input) if delay_input else 0.5
                
                # Build list of samples to test: (class_idx, sample_idx)
                test_queue = []
                if mode_choice in [0, 1, 2]:
                    # Specific class
                    samples_avail = len(X_by_label[mode_choice])
                    indices = random.sample(range(samples_avail), min(count, samples_avail))
                    for idx in indices:
                        test_queue.append((mode_choice, idx))
                else:
                    # Mixed classes
                    for _ in range(count):
                        c_idx = random.choice([0, 1, 2])
                        s_idx = random.randint(0, len(X_by_label[c_idx]) - 1)
                        test_queue.append((c_idx, s_idx))
                
                print(f"\nStarting batch test of {len(test_queue)} samples...")
                print("Press Ctrl+C to abort batch run.")
                
                results = []  # List of dicts: {gt, pred, success, latency}
                
                for step, (gt_class, sample_idx) in enumerate(test_queue):
                    print(f"\n[{step+1}/{len(test_queue)}] Sending {LABEL_MAP[gt_class]} sample #{sample_idx}...")
                    
                    # Select 1 complex sample: shape (50, 114)
                    complex_data = X_by_label[gt_class][sample_idx]  # Shape: (50, 114) complex
                    
                    # Serialize raw complex data (interleave imaginary and real parts)
                    raw_floats = np.empty((50, 228), dtype=np.float32)
                    raw_floats[:, 0::2] = complex_data.imag
                    raw_floats[:, 1::2] = complex_data.real
                    flat_features = raw_floats.flatten()
                    
                    # Framing
                    header = bytes([0xAA, 0xBB, 0xCC, 0xDD])
                    data_bytes = struct.pack('<11400f', *flat_features)
                    length_bytes = struct.pack('<I', len(data_bytes))
                    checksum = sum(data_bytes) & 0xFFFFFFFF
                    checksum_bytes = struct.pack('<I', checksum)
                    packet = header + length_bytes + data_bytes + checksum_bytes
                    
                    # Flush and Write
                    ser.reset_input_buffer()
                    ser.write(packet)
                    
                    # Listen for response
                    pred_class = -1
                    latency_us = 0
                    success = False
                    
                    start_time = time.time()
                    while time.time() - start_time < 5.0:
                        if ser.in_waiting > 0:
                            try:
                                line = ser.readline().decode('utf-8', errors='ignore').strip()
                                if line:
                                    print(f"  [Board Log] {line}")
                                    if "Inference Result:" in line:
                                        success = True
                                        # Parse prediction class
                                        if "Wave/Cut" in line or "挥手" in line:
                                            pred_class = 0
                                        elif "Grip" in line or "抓握" in line:
                                            pred_class = 1
                                        elif "Circle/Draw_o" in line or "画圈" in line:
                                            pred_class = 2
                                        elif "Unknown" in line or "未知" in line:
                                            pred_class = 3
                                            
                                        # Parse latency
                                        time_match = re.search(r'Time taken:\s*(\d+)\s*us', line)
                                        if time_match:
                                            latency_us = int(time_match.group(1))
                                        break
                                    elif "Inference failed" in line:
                                        break
                            except Exception as read_err:
                                print(f"  Error reading logs: {read_err}")
                                break
                        time.sleep(0.01)
                    
                    correct = (pred_class == gt_class) if success else False
                    results.append({
                        'gt': gt_class,
                        'pred': pred_class,
                        'success': success,
                        'correct': correct,
                        'latency': latency_us
                    })
                    
                    if success:
                        status_str = "\033[92mCORRECT\033[0m" if correct else f"\033[91mWRONG\033[0m (Predicted: {LABEL_MAP.get(pred_class, 'unknown')})"
                        print(f"  >> Outcome: {status_str} | Latency: {latency_us / 1000.0:.2f} ms")
                    else:
                        print("  >> Outcome: \033[91mTIMEOUT/FAILED\033[0m")
                        
                    time.sleep(delay)
                    
                # Print batch summary report
                print("\n" + "="*52)
                print("              BATCH TEST SWEEP REPORT               ")
                print("" + "="*52)
                total_sent = len(results)
                successful = sum(1 for r in results if r['success'])
                correct = sum(1 for r in results if r['correct'])
                
                accuracy = (correct / total_sent * 100.0) if total_sent > 0 else 0
                avg_latency = (sum(r['latency'] for r in results if r['success']) / successful / 1000.0) if successful > 0 else 0
                
                print(f"Total Samples Sent:      {total_sent}")
                print(f"Successful Inferences:   {successful}")
                print(f"Correct Predictions:     {correct}")
                print(f"Overall Accuracy:        \033[93m{accuracy:.2f}%\033[0m")
                print(f"Average Inference Time:  {avg_latency:.2f} ms")
                print("-"*52)
                
                # Print per-class details
                for c_idx, c_name in LABEL_MAP.items():
                    class_results = [r for r in results if r['gt'] == c_idx]
                    class_total = len(class_results)
                    class_correct = sum(1 for r in class_results if r['correct'])
                    class_acc = (class_correct / class_total * 100.0) if class_total > 0 else 0
                    if class_total > 0:
                        print(f"Class '{c_name}': Accuracy = {class_acc:.2f}% ({class_correct}/{class_total})")
                print("="*52)
                
            except KeyboardInterrupt:
                print("\nBatch test sweep aborted by user.")
            except ValueError:
                print("Invalid setting input.")
                
        elif opt == '3':
            print("Closing port and exiting...")
            break
        else:
            print("Invalid option.")

    ser.close()

if __name__ == '__main__':
    main()
