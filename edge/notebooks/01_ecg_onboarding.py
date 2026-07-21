"""
ECG Onboarding — Week 1 Exploration Script
===========================================
Taher KHALLAF — Volet Edge

This script covers the ECG onboarding topics:
  - Understanding 12-lead ECG signals
  - Exploring the Chapman-Shaoxing dataset (WFDB format)
  - Visualizing signals per lead
  - Understanding SNOMED-CT diagnostic labels
  - Class distribution analysis
  - AAMI rhythm classes overview
  - Computing baseline metrics (macro F1 explanation)

Run: python notebooks/01_ecg_onboarding.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
import wfdb
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_loader import load_config, parse_snomed_labels, scan_dataset


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def explore_ecg_basics():
    """Part 1: ECG fundamentals overview."""
    print_section("PART 1 — ECG Fundamentals")

    info = """
    A 12-lead ECG records the heart's electrical activity from 12 different
    angles (leads). Each lead captures a different "view" of the heart:

    LIMB LEADS (frontal plane):
      I   — Right arm → Left arm
      II  — Right arm → Left leg
      III — Left arm  → Left leg

    AUGMENTED LIMB LEADS (frontal plane):
      aVR — Augmented Vector Right
      aVL — Augmented Vector Left
      aVF — Augmented Vector Foot

    PRECORDIAL LEADS (horizontal plane):
      V1–V6 — Across the chest (anterior → lateral)

    KEY WAVEFORM COMPONENTS:
      P wave   — Atrial depolarization (contraction)
      QRS      — Ventricular depolarization
      T wave   — Ventricular repolarization
      PR       — Atrio-ventricular conduction time
      QT       — Total ventricular activity duration

    AAMI CLASSES (Association for the Advancement of Medical Instrumentation):
      N — Normal / Sinus Rhythm
      S — Supraventricular ectopic
      V — Ventricular ectopic
      F — Fusion
      Q — Unknown / Paced

    RHYTHM vs BEAT CLASSIFICATION:
      - Beat-level: classify individual heartbeats (MIT-BIH style)
      - Rhythm-level: classify the overall heart rhythm from a segment
        (Chapman-Shaoxing is rhythm-level: 10s recordings)

    EVALUATION METRIC — Macro F1:
      F1 per class = 2 * (Precision * Recall) / (Precision + Recall)
      Macro F1 = average of F1 scores across all classes
      → Treats each class equally regardless of support (important for
        imbalanced ECG datasets where Sinus Rhythm dominates)
    """
    print(info)


def explore_dataset(config: dict):
    """Part 2: Explore the Chapman-Shaoxing dataset."""
    print_section("PART 2 — Chapman-Shaoxing Dataset Exploration")

    data_dir = config['data']['raw_dir']

    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Run 'python scripts/extract_dataset.py' first!")
        return None

    # Count files
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)
    mat_files = glob.glob(os.path.join(data_dir, '**', '*.mat'), recursive=True)
    print(f"  .hea (header) files:  {len(hea_files)}")
    print(f"  .mat (signal) files:  {len(mat_files)}")
    print(f"  Total records:        {len(hea_files)}")

    # Load a sample record
    if not hea_files:
        print("No records found!")
        return None

    sample_path = os.path.splitext(hea_files[0])[0]
    record = wfdb.rdrecord(sample_path)

    print(f"\n  Sample Record: {os.path.basename(sample_path)}")
    print(f"  Sampling frequency:   {record.fs} Hz")
    print(f"  Number of leads:      {record.n_sig}")
    print(f"  Signal length:        {record.sig_len} samples ({record.sig_len/record.fs:.1f}s)")
    print(f"  Lead names:           {record.sig_name}")
    print(f"  Physical units:       {record.units}")
    print(f"  Signal shape:         {record.p_signal.shape}")

    # Header comments (contain diagnosis codes)
    header = wfdb.rdheader(sample_path)
    print(f"\n  Header comments:")
    for comment in header.comments:
        print(f"    {comment}")

    return hea_files


def visualize_sample_ecg(config: dict):
    """Part 3: Visualize a sample 12-lead ECG."""
    print_section("PART 3 — ECG Signal Visualization")

    data_dir = config['data']['raw_dir']
    hea_files = glob.glob(os.path.join(data_dir, '**', '*.hea'), recursive=True)

    if not hea_files:
        print("No records found. Extract dataset first.")
        return

    # Read a sample record
    sample_path = os.path.splitext(hea_files[0])[0]
    record = wfdb.rdrecord(sample_path)
    signal = record.p_signal  # (5000, 12)
    lead_names = record.sig_name
    fs = record.fs

    # Parse its diagnosis
    header = wfdb.rdheader(sample_path)
    label_mapping = config.get('label_mapping', {})
    labels = parse_snomed_labels(header, label_mapping)
    diagnosis = ', '.join(labels) if labels else 'Unknown'

    # Create time axis in seconds
    time_axis = np.arange(signal.shape[0]) / fs

    # --- Plot 1: All 12 leads ---
    os.makedirs('reports', exist_ok=True)

    fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)
    fig.suptitle(f'12-Lead ECG — {os.path.basename(sample_path)}\nDiagnosis: {diagnosis}',
                 fontsize=14, fontweight='bold')

    for i, (ax, name) in enumerate(zip(axes, lead_names)):
        ax.plot(time_axis, signal[:, i], linewidth=0.6, color='#1a73e8')
        ax.set_ylabel(name, fontsize=10, rotation=0, labelpad=30)
        ax.set_xlim(0, time_axis[-1])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='y', labelsize=7)

    axes[-1].set_xlabel('Time (seconds)', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/sample_12lead_ecg.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reports/sample_12lead_ecg.png")

    # --- Plot 2: Zoom on Lead II (most clinically used) ---
    fig, ax = plt.subplots(figsize=(14, 4))
    lead_ii_idx = lead_names.index('II') if 'II' in lead_names else 1
    # Show first 3 seconds (zoom)
    zoom_samples = int(3 * fs)
    ax.plot(time_axis[:zoom_samples], signal[:zoom_samples, lead_ii_idx],
            linewidth=1.0, color='#d32f2f')
    ax.set_title(f'Lead II — Zoomed (first 3 seconds) — {diagnosis}', fontsize=13)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/sample_lead_ii_zoom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reports/sample_lead_ii_zoom.png")

    # --- Plot 3: Signal statistics per lead ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    means = [signal[:, i].mean() for i in range(signal.shape[1])]
    stds = [signal[:, i].std() for i in range(signal.shape[1])]
    ranges = [signal[:, i].max() - signal[:, i].min() for i in range(signal.shape[1])]

    axes[0].bar(lead_names, means, color='#1a73e8', alpha=0.8)
    axes[0].set_title('Mean Amplitude per Lead')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].bar(lead_names, stds, color='#34a853', alpha=0.8)
    axes[1].set_title('Std Deviation per Lead')
    axes[1].tick_params(axis='x', rotation=45)

    axes[2].bar(lead_names, ranges, color='#ea4335', alpha=0.8)
    axes[2].set_title('Signal Range per Lead')
    axes[2].tick_params(axis='x', rotation=45)

    plt.suptitle(f'Signal Statistics — {os.path.basename(sample_path)}', fontsize=13)
    plt.tight_layout()
    plt.savefig('reports/signal_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: reports/signal_statistics.png")


def analyze_class_distribution(config: dict):
    """Part 4: Full class distribution analysis."""
    print_section("PART 4 — Class Distribution Analysis")

    data_dir = config['data']['raw_dir']
    label_mapping = config.get('label_mapping', {})

    if not os.path.exists(data_dir):
        print("Data directory not found. Extract dataset first.")
        return

    # Scan entire dataset
    print("  Scanning all records (this may take a few minutes)...\n")
    df = scan_dataset(data_dir, label_mapping)

    if df.empty:
        print("No labeled records found!")
        return

    print(f"\n  Total labeled records: {len(df)}")

    # Class distribution
    class_counts = df['primary_label'].value_counts()
    print(f"\n  Class Distribution (primary label):")
    print(f"  {'Class':<35} {'Count':>6} {'Percentage':>10}")
    print(f"  {'-'*55}")
    for cls, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls:<35} {count:>6} {pct:>9.1f}%")

    # Top 4 classes (what we'll use for training)
    num_classes = config.get('model', {}).get('num_classes', 4)
    top_classes = class_counts.nlargest(num_classes)
    print(f"\n  Top {num_classes} classes (used for training):")
    for cls, count in top_classes.items():
        print(f"    {cls}: {count} records")

    # Plot distribution
    os.makedirs('reports', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # All classes
    colors = sns.color_palette('viridis', len(class_counts))
    axes[0].barh(class_counts.index, class_counts.values, color=colors)
    axes[0].set_title('All Classes Distribution', fontsize=13)
    axes[0].set_xlabel('Number of Records')
    for i, v in enumerate(class_counts.values):
        axes[0].text(v + 20, i, str(v), va='center', fontsize=9)

    # Top N classes
    colors_top = sns.color_palette('Set2', len(top_classes))
    axes[1].bar(top_classes.index, top_classes.values, color=colors_top)
    axes[1].set_title(f'Top {num_classes} Classes (Training Set)', fontsize=13)
    axes[1].set_ylabel('Number of Records')
    axes[1].tick_params(axis='x', rotation=30)
    for i, v in enumerate(top_classes.values):
        axes[1].text(i, v + 20, str(v), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('reports/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: reports/class_distribution.png")

    # Multi-label analysis
    multi_label_count = df['labels'].apply(len)
    print(f"\n  Multi-label statistics:")
    print(f"    Records with 1 label:  {(multi_label_count == 1).sum()}")
    print(f"    Records with 2 labels: {(multi_label_count == 2).sum()}")
    print(f"    Records with 3+ labels: {(multi_label_count >= 3).sum()}")

    return df


def test_model_architecture():
    """Part 5: Verify the model works with a dummy input."""
    print_section("PART 5 — Model Architecture Verification")

    from src.model import build_model, count_parameters
    import torch

    config = load_config('configs/config.yaml')
    model = build_model(config)

    print(f"  Model: {config['model']['name']}")
    print(f"  Trainable parameters: {count_parameters(model):,}")
    print(f"\n  Architecture:")
    print(f"  {model}")

    # Dummy forward pass
    x = torch.randn(2, 12, 5000)  # batch of 2
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"\n  Input shape:  {x.shape}  → (batch=2, leads=12, samples=5000)")
    print(f"  Output shape: {out.shape} → (batch=2, classes={config['model']['num_classes']})")
    print(f"\n  ✅ Model forward pass successful!")


def test_dataloader(config: dict):
    """Part 6: Test that the full data pipeline works end-to-end."""
    print_section("PART 6 — DataLoader End-to-End Test")

    from src.data_loader import create_dataloaders

    train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(config)

    print(f"\n  Class mapping: {class_to_idx}")
    print(f"\n  Loading first batch...")

    for batch_signals, batch_labels in train_loader:
        print(f"  Batch signals shape: {batch_signals.shape}")
        print(f"  Batch labels shape:  {batch_labels.shape}")
        print(f"  Batch labels:        {batch_labels.tolist()}")
        print(f"  Signal dtype:        {batch_signals.dtype}")
        print(f"  Signal range:        [{batch_signals.min():.4f}, {batch_signals.max():.4f}]")
        print(f"\n  ✅ DataLoader works correctly!")
        break


# ============================================================
#  MAIN — Run all onboarding steps
# ============================================================
if __name__ == '__main__':
    print("\n" + "🫀"*30)
    print("  ECG ONBOARDING — WEEK 1")
    print("  Taher KHALLAF — Volet Edge")
    print("🫀"*30)

    config = load_config('configs/config.yaml')

    # Part 1: ECG theory (always runs)
    explore_ecg_basics()

    # Part 2: Dataset exploration
    hea_files = explore_dataset(config)

    if hea_files:
        # Part 3: Visualization
        visualize_sample_ecg(config)

        # Part 4: Class distribution
        analyze_class_distribution(config)

        # Part 5: Model architecture check
        test_model_architecture()

        # Part 6: DataLoader test
        test_dataloader(config)
    else:
        print("\n⚠ Skipping Parts 3-6: extract the dataset first!")
        print("  Run: python scripts/extract_dataset.py")

    print_section("ONBOARDING COMPLETE")
    print("  Next steps:")
    print("    1. Review the plots in reports/")
    print("    2. Run training: python -m src.train")
    print("    3. Export to ONNX: python -m src.export_onnx --model-path models/best_model.pth")
    print()
