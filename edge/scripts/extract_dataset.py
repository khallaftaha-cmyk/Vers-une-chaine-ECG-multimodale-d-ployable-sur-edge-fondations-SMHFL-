import zipfile
import argparse
import yaml
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Extract ECG dataset")
    parser.add_argument("--zip_path", type=str, help="Path to dataset ZIP file")
    args = parser.parse_args()

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        config = load_config(config_path)
    else:
        config = {"data": {"dataset_zip": "", "raw_dir": "data/raw/WFDB_ChapmanShaoxing"}}

    zip_path = args.zip_path if args.zip_path else config.get("data", {}).get("dataset_zip")
    if not zip_path:
        print("Please provide a zip_path via CLI or config.")
        return

    zip_file = Path(zip_path)
    extract_dir = Path(config.get("data", {}).get("raw_dir", "data/raw/WFDB_ChapmanShaoxing"))

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
