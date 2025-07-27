import sys
import os
import argparse
import json
from src.detector import load_model_and_artifacts, predict_headings, create_json_outline

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    parser = argparse.ArgumentParser(
        description="""
        Connect the Dots: PDF Title and Heading Extractor.
        Processes a single PDF or a directory of PDFs to extract a structured
        JSON outline of the document's title and headings (H1, H2, H3).
        """
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="The path to a single PDF file or a directory containing PDF files."
    )
    args = parser.parse_args()

    input_path = args.input_path
    
    if os.path.exists("/app/input"):
        output_dir = "/app/output"
    else:
        output_dir = "output"

    if not os.path.exists(input_path):
        print(f"Error: The path '{input_path}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    pdf_files = []
    if os.path.isdir(input_path):
        print(f"Searching for PDF files in directory: {input_path}")
        pdf_files = sorted([os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith('.pdf')])
    elif os.path.isfile(input_path) and input_path.lower().endswith('.pdf'):
        pdf_files.append(input_path)
    
    if not pdf_files:
        print("No PDF files found to process.")
        sys.exit(0)

    model, features, label_encoder = load_model_and_artifacts()

    if not all([model, features, label_encoder]):
        print("Failed to load model and artifacts. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nFound {len(pdf_files)} PDF(s) to process.")

    for pdf_file in pdf_files:
        print(f"--- Processing: {os.path.basename(pdf_file)} ---")
        output_filename = os.path.splitext(os.path.basename(pdf_file))[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)

        try:
            headings_df = predict_headings(pdf_file, model, features, label_encoder)
            
            if not headings_df.empty:
                # The detector script returns a JSON string
                json_output_string = create_json_outline(headings_df, os.path.basename(pdf_file))
                with open(output_path, 'w') as f:
                    f.write(json_output_string)
                print(f"  -> Saved output to {output_path}")
            else:
                # Create a JSON with an empty outline if no headings were found
                json_output = {"title": os.path.basename(pdf_file), "outline": []}
                with open(output_path, 'w') as f:
                    json.dump(json_output, f, indent=2)
                print(f"  -> No headings found. Saved empty outline to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {os.path.basename(pdf_file)}: {e}")
            with open(output_path, 'w') as f:
                json.dump({"error": str(e), "file": os.path.basename(pdf_file)}, f, indent=2)
            print(f"  -> Saved error log to {output_path}")

if __name__ == "__main__":
    main()

