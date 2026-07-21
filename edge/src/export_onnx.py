"""Export trained PyTorch model to ONNX format."""

import os
import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
import yaml
from pathlib import Path

from edge.src.model import build_model
from edge.src.data_loader import load_config

def export_to_onnx(model_path: str, config_path: str = 'configs/config.yaml', output_path: str = None):
    # 1. Load config
    config = load_config(config_path)
    
    if output_path is None:
        output_path = config.get('onnx', {}).get('fp32_path', 'models/ecg_model_fp32.onnx')
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    # 2. Build model architecture
    model = build_model(config)
    
    # 3. Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # 4. Set eval mode
    model.eval()
    
    # 5. Dummy input
    dummy_input = torch.randn(1, 12, 5000)
    
    # 6. Export
    opset_version = config.get('onnx', {}).get('opset_version', 17)
    dynamic_axes_cfg = config.get('onnx', {}).get('dynamic_axes', True)
    
    dynamic_axes = None
    if dynamic_axes_cfg:
        dynamic_axes = {
            'ecg_input': {0: 'batch_size'},
            'classification': {0: 'batch_size'}
        }
        
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['ecg_input'],
        output_names=['classification'],
        dynamic_axes=dynamic_axes
    )
    
    # 7. Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    # 8. Size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"ONNX Model Size: {size_mb:.2f} MB")
    
    # 9. Verify with ONNX Runtime
    session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outs = session.run(None, ort_inputs)
    
    with torch.no_grad():
        torch_out = model(dummy_input)
        
    np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-03, atol=1e-05)
    print("Verification passed: ONNX output matches PyTorch output.")
    
    # 10. Success
    print(f"Successfully exported model to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .pth file")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX path")
    
    args = parser.parse_args()
    export_to_onnx(args.model_path, args.config, args.output)
