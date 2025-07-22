import os
import sys
import pandas as pd
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.parser import extract_features
except ImportError as e:
    print(f"couldnt not import from 'src.parser': {e}\nrun in root dir")
    sys.exit(1)

def create_labeling_file(pdf_path: str, output_dir: str):
    pdf_filename = os.path.basename(pdf_path)
    csv_filename = os.path.splitext(pdf_filename)[0] + ".csv"
    output_path = os.path.join(output_dir, csv_filename)

    if os.path.exists(output_path):
        print(f"skipping '{pdf_filename}', labeling file already exists: {output_path}")
        return
    

    print(f"\nprocessing '{pdf_filename}' for labeling...")
    try:
        features_df = extract_features(pdf_path)
    except Exception as e:
        print(f"couldnt extract features from '{pdf_filename}': {e}")
        return

    if features_df.empty:
        print(f"no features in '{pdf_filename}'. skipping.")
        return

    features_df['label'] = ''
    cols = features_df.columns.tolist()
    cols.insert(1, cols.pop(cols.index('label')))
    features_df = features_df[cols]

    print(f"saving features to '{output_path}' for labeling.")
    features_df.to_csv(output_path, index=False, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Generate feature files for manual labeling from PDFs.")
    parser.add_argument(
        'filenames', 
        nargs='*', 
        help="Specific PDF filenames to process from the 'data/raw_pdfs' directory. If empty, all PDFs will be processed."
    )
    args = parser.parse_args()

    raw_pdfs_dir = "data/raw_pdfs"
    labeled_data_dir = "data/labeled_data"

    os.makedirs(raw_pdfs_dir, exist_ok=True)
    os.makedirs(labeled_data_dir, exist_ok=True)

    if args.filenames:
        pdf_files_to_process = [f for f in args.filenames if f.endswith(".pdf")]
    else:
        print("no specific files provided. processing all PDFs in 'data/raw_pdfs/'...")
        pdf_files_to_process = [f for f in os.listdir(raw_pdfs_dir) if f.endswith(".pdf")]

    if not pdf_files_to_process:
        print("no PDF files found to process.")
        print(f"please add PDFs to the '{raw_pdfs_dir}' directory.")
        return

    for pdf_filename in tqdm(pdf_files_to_process, desc="Generating labeling files"):
        pdf_path = os.path.join(raw_pdfs_dir, pdf_filename)
        if not os.path.exists(pdf_path):
            print(f"file not found '{pdf_path}'. skipping.")
            continue
        create_labeling_file(pdf_path, labeled_data_dir)

    print("\n--- labeling file generation complete ---")
if __name__ == "__main__":
    main()
