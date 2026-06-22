import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add train_src to system path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_src"))
from models import Advanced1DCNN
from dataset import load_dataset, preprocess_csi_fusion

from esp_ppq.api import espdl_quantize_onnx

def main():
    # Configure console to output UTF-8 to prevent encoding crashes on Windows with emojis
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Fallback for older python where reconfigure is not available
        
    transform_tool_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(transform_tool_dir, "models", "best_optimized_cnn.pth")
    onnx_path = os.path.join(transform_tool_dir, "cnn_model.onnx")
    espdl_path = os.path.join(transform_tool_dir, "cnn_model.espdl")
    dataset_dir = os.path.join(transform_tool_dir, "dataset", "dataset_2026_6_10")
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Instantiate Advanced1DCNN
    model = Advanced1DCNN(n_subcarriers=228, num_classes=4)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 1. Export to ONNX
    dummy_input = torch.randn(1, 228, 50)
    print(f"Exporting model to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully.")
    
    # 2. Load calibration dataset
    print(f"Loading calibration dataset from {dataset_dir}...")
    X_by_label = load_dataset(dataset_dir)
    
    # Accumulate samples for calibration
    calib_samples = []
    # Collect up to 32 samples spread across classes
    samples_per_class = 12
    for label_idx, data_complex in X_by_label.items():
        # data_complex shape: (N, 50, 114)
        N = len(data_complex)
        select_count = min(N, samples_per_class)
        indices = np.linspace(0, N - 1, select_count, dtype=int)
        class_samples = data_complex[indices]
        
        # Preprocess using csi fusion
        preprocessed = preprocess_csi_fusion(class_samples)  # (select_count, 50, 228)
        
        for sample in preprocessed:
            # sample shape: (50, 228)
            # We need shape (1, 228, 50) as PyTorch tensor
            t_sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)
            calib_samples.append(t_sample)
            
    print(f"Total calibration samples collected: {len(calib_samples)}")
    
    # Truncate or pad to exactly 32 samples if needed
    if len(calib_samples) < 32:
        # Pad with duplicate samples
        while len(calib_samples) < 32:
            calib_samples.append(calib_samples[0])
    calib_samples = calib_samples[:32]
    
    def collate_fn(batch):
        return batch.to("cpu")
        
    # 3. Quantize using esp-ppq
    print(f"Running esp-ppq quantization to generate {espdl_path}...")
    espdl_quantize_onnx(
        onnx_import_file=onnx_path,
        espdl_export_file=espdl_path,
        calib_dataloader=calib_samples,
        calib_steps=32,
        input_shape=[1, 228, 50],
        target="esp32p4",
        num_of_bits=8,
        collate_fn=collate_fn,
        device="cpu",
        error_report=True,
        export_test_values=True,
        verbose=1
    )
    print("esp-ppq Quantization completed successfully.")

if __name__ == '__main__':
    main()
