"""Data loader for Chapman-Shaoxing 12-lead ECG dataset (WFDB format)."""

import os
import glob
import numpy as np
import pandas as pd
import wfdb
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm

def load_config(config_path='configs/config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def parse_snomed_labels(header: wfdb.Record, label_mapping: dict) -> list[str]:
    labels = []
    if not header.comments:
        return labels
    for comment in header.comments:
        if comment.startswith('Dx:'):
            codes = comment.replace('Dx:', '').strip().split(',')
            for code in codes:
                code = code.strip()
                if code in label_mapping:
                    labels.append(label_mapping[code])
    return labels

def scan_dataset(data_dir: str, label_mapping: dict) -> pd.DataFrame:
    records = []
    # Search for all .hea files
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
    
    for hea_path in tqdm(hea_files, desc="Scanning dataset"):
        # We need the record name which is the hea file without extension
        record_path = os.path.splitext(hea_path)[0]
        record_name = os.path.basename(record_path)
        
        try:
            header = wfdb.rdheader(record_path)
            labels = parse_snomed_labels(header, label_mapping)
            if labels:
                primary_label = labels[0]
                records.append({
                    'record_name': record_name,
                    'file_path': record_path,
                    'labels': labels,
                    'primary_label': primary_label
                })
        except Exception as e:
            print(f"Error reading header {hea_path}: {e}")
            
    df = pd.DataFrame(records)
    if not df.empty:
        print("\nClass distribution:")
        print(df['primary_label'].value_counts())
    return df

def filter_top_classes(df: pd.DataFrame, num_classes: int = 4) -> Tuple[pd.DataFrame, dict]:
    top_classes = df['primary_label'].value_counts().nlargest(num_classes).index.tolist()
    filtered_df = df[df['primary_label'].isin(top_classes)].copy()
    
    class_to_idx = {cls: idx for idx, cls in enumerate(top_classes)}
    return filtered_df, class_to_idx

class ChapmanECGDataset(Dataset):
    def __init__(self, records_df: pd.DataFrame, data_dir: str, class_to_idx: dict, transform=None, sequence_length=5000, num_leads=12):
        self.records_df = records_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.sequence_length = sequence_length
        self.num_leads = num_leads

    def __len__(self) -> int:
        return len(self.records_df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        row = self.records_df.iloc[idx]
        record_path = row['file_path']
        
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal # shape (5000, 12) typically
        
        # Replace NaNs with 0
        signal = np.nan_to_num(signal)
        
        # Pad or truncate if needed (along axis 0)
        current_len = signal.shape[0]
        if current_len < self.sequence_length:
            pad_width = ((0, self.sequence_length - current_len), (0, 0))
            signal = np.pad(signal, pad_width, mode='constant', constant_values=0)
        elif current_len > self.sequence_length:
            signal = signal[:self.sequence_length, :]
            
        # Transpose to (12, 5000) for Conv1d
        signal = signal.T
        
        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        
        if self.transform:
            signal_tensor = self.transform(signal_tensor)
            
        label_index = self.class_to_idx[row['primary_label']]
        
        return signal_tensor, label_index

def create_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    data_dir = config.get('data', {}).get('raw_dir', 'data/raw')
    label_mapping = config.get('label_mapping', {})
    num_classes = config.get('model', {}).get('num_classes', 4)
    
    df = scan_dataset(data_dir, label_mapping)
    df, class_to_idx = filter_top_classes(df, num_classes)
    
    train_split = config.get('training', {}).get('train_split', 0.7)
    val_split = config.get('training', {}).get('val_split', 0.15)
    test_split = config.get('training', {}).get('test_split', 0.15)
    
    seed = config.get('training', {}).get('seed', 42)
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 4)
    seq_length = config.get('data', {}).get('sequence_length', 5000)
    
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_split), stratify=df['primary_label'], random_state=seed
    )
    
    val_ratio = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio, stratify=temp_df['primary_label'], random_state=seed
    )
    
    train_dataset = ChapmanECGDataset(train_df, data_dir, class_to_idx, sequence_length=seq_length)
    val_dataset = ChapmanECGDataset(val_df, data_dir, class_to_idx, sequence_length=seq_length)
    test_dataset = ChapmanECGDataset(test_df, data_dir, class_to_idx, sequence_length=seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, class_to_idx

if __name__ == '__main__':
    # Assume config exists at configs/config.yaml for testing
    try:
        config = load_config('configs/config.yaml')
        train_loader, val_loader, test_loader, class_idx = create_dataloaders(config)
        for batch_signals, batch_labels in train_loader:
            print(f"Sample batch shape: {batch_signals.shape}, labels: {batch_labels.shape}")
            break
    except FileNotFoundError:
        print("Config file not found for testing. Please ensure configs/config.yaml exists.")
