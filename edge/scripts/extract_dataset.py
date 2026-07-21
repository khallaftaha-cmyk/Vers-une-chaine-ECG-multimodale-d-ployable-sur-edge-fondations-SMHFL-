import sys
import zipfile
import argparse
import yaml
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../edge
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'


def load_config(config_path=None):
    config_path = config_path or DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Extract ECG dataset")
    parser.add_argument("--zip_path", type=str, help="Path to dataset ZIP file")
    args = parser.parse_args()

    if DEFAULT_CONFIG_PATH.exists():
        config = load_config()
    else:
        config = {"data": {"dataset_zip": "", "raw_dir": "data/raw/WFDB_ChapmanShaoxing"}}

    zip_path = args.zip_path if args.zip_path else config.get("data", {}).get("dataset_zip")
    if not zip_path:
        print("Please provide a zip_path via CLI or config.")
        return

    zip_file = Path(zip_path)

    raw_dir = config.get("data", {}).get("raw_dir", "data/raw/WFDB_ChapmanShaoxing")
    extract_dir = Path(raw_dir)
    if not extract_dir.is_absolute():
        extract_dir = PROJECT_ROOT / extract_dir

    if not zip_file.exists():
        print(f"ZIP file not found at {zip_file}")
        return

    print(f"Extracting {zip_file} to {extract_dir}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("Extraction complete.")

    hea_count = len(list(extract_dir.rglob("*.hea")))
    mat_count = len(list(extract_dir.rglob("*.mat")))

    print(f"Summary: Found {hea_count} .hea files and {mat_count} .mat files.")
    print(f"Total records found (assuming 1 .hea per record): {hea_count}")


if __name__ == "__main__":
    main()