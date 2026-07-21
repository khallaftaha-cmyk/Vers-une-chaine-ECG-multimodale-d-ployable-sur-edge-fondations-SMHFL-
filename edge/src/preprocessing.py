"""ECG signal preprocessing: filtering, resampling, normalization."""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional
import torch

def bandpass_filter(ecg_signal: np.ndarray, lowcut=0.5, highcut=45.0, fs=500, order=4) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    
    filtered_signal = np.zeros_like(ecg_signal)
    if len(ecg_signal.shape) > 1:
        for i in range(ecg_signal.shape[1]):
            filtered_signal[:, i] = scipy_signal.filtfilt(b, a, ecg_signal[:, i])
    else:
        filtered_signal = scipy_signal.filtfilt(b, a, ecg_signal)
        
    return filtered_signal

def normalize_signal(ecg_signal: np.ndarray, method='zscore') -> np.ndarray:
    normalized = np.zeros_like(ecg_signal, dtype=np.float64)
    if len(ecg_signal.shape) > 1:
        for i in range(ecg_signal.shape[1]):
            col = ecg_signal[:, i]
            if method == 'zscore':
                std = np.std(col)
                if std > 0:
                    normalized[:, i] = (col - np.mean(col)) / std
                else:
                    normalized[:, i] = col - np.mean(col)
            elif method == 'minmax':
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val > min_val:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                else:
                    normalized[:, i] = col - min_val
    else:
        if method == 'zscore':
            std = np.std(ecg_signal)
            normalized = (ecg_signal - np.mean(ecg_signal)) / std if std > 0 else ecg_signal - np.mean(ecg_signal)
        elif method == 'minmax':
            min_val = np.min(ecg_signal)
            max_val = np.max(ecg_signal)
            normalized = (ecg_signal - min_val) / (max_val - min_val) if max_val > min_val else ecg_signal - min_val
            
    return normalized

def resample_signal(ecg_signal: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    num_samples = ecg_signal.shape[0]
    target_samples = int(num_samples * target_fs / original_fs)
    resampled = scipy_signal.resample(ecg_signal, target_samples, axis=0)
    return resampled

def pad_or_truncate(ecg_signal: np.ndarray, target_length: int) -> np.ndarray:
    current_length = ecg_signal.shape[0]
    if current_length < target_length:
        if len(ecg_signal.shape) > 1:
            pad_width = ((0, target_length - current_length), (0, 0))
        else:
            pad_width = (0, target_length - current_length)
        padded = np.pad(ecg_signal, pad_width, mode='constant', constant_values=0)
        return padded
    elif current_length > target_length:
        return ecg_signal[:target_length]
    return ecg_signal

class ECGPreprocessor:
    def __init__(self, fs=500, lowcut=0.5, highcut=45.0, normalize_method='zscore', target_length=5000):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.normalize_method = normalize_method
        self.target_length = target_length

    def __call__(self, ecg_signal: np.ndarray) -> np.ndarray:
        # Bandpass filter
        processed = bandpass_filter(ecg_signal, lowcut=self.lowcut, highcut=self.highcut, fs=self.fs)
        # Normalize
        processed = normalize_signal(processed, method=self.normalize_method)
        # Pad or truncate
        processed = pad_or_truncate(processed, self.target_length)
        return processed

class ECGTransform:
    def __init__(self, preprocessor: ECGPreprocessor):
        self.preprocessor = preprocessor
        
    def __call__(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        # Convert to numpy and transpose to (sequence, leads)
        signal_np = signal_tensor.numpy().T
        
        # Process
        processed_np = self.preprocessor(signal_np)
        
        # Transpose back to (leads, sequence) and convert to tensor
        processed_tensor = torch.tensor(processed_np.T, dtype=torch.float32)
        return processed_tensor

if __name__ == '__main__':
    # Create dummy 12-lead signal (6000 samples)
    dummy_signal = np.random.randn(6000, 12)
    print("Original shape:", dummy_signal.shape)
    print("Original mean (lead 0):", np.mean(dummy_signal[:, 0]))
    print("Original std (lead 0):", np.std(dummy_signal[:, 0]))
    
    preprocessor = ECGPreprocessor()
    processed_signal = preprocessor(dummy_signal)
    
    print("Processed shape:", processed_signal.shape)
    print("Processed mean (lead 0):", np.mean(processed_signal[:, 0]))
    print("Processed std (lead 0):", np.std(processed_signal[:, 0]))
