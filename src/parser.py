import fitz
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from typing import List, Dict, Any
from tqdm import tqdm

try:
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    print("Please ensure you have an internet connection for the first download.")
    EMBEDDING_MODEL = None

def get_font_statistics(doc: fitz.Document) -> Dict[str, float]:
    sizes = {}
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for l in b["lines"]:
                    for s in l["spans"]:
                        size = round(s["size"])
                        sizes[size] = sizes.get(size, 0) + len(s["text"].strip())
    if not sizes:
        return {"avg_size": 12.0, "common_size": 12.0}
    avg_size = sum(s * c for s, c in sizes.items()) / sum(sizes.values())
    common_size = max(sizes, key=lambda k: sizes.get(k, 0))
    return {"avg_size": avg_size, "common_size": common_size}

def get_toc_match_score(text: str, page_num: int, toc: List) -> float:
    if not toc:
        return 0.0
    
    max_score = 0
    for level, title, page in toc:
        if abs(page - page_num) <= 1:
            score = fuzz.ratio(text, title)
            if score > max_score:
                max_score = score
    return max_score / 100.0 

def extract_features(pdf_path: str) -> pd.DataFrame:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return pd.DataFrame()

    font_stats = get_font_statistics(doc)
    doc_common_font_size = font_stats["common_size"]
    toc = doc.get_toc()

    features = []
    all_span_texts = []

    # First pass: collect all span information
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        blocks = page.get_text("dict")["blocks"]
        
        prev_span_bbox_on_page = None
        for block_num, b in enumerate(blocks):
            if b['type'] == 0:
                for line_num, l in enumerate(b["lines"]):
                    for span_num, s in enumerate(l["spans"]):
                        text = s["text"].strip()
                        if not text:
                            continue

                        font_size = s["size"]
                        
                        space_to_prev = 0
                        if prev_span_bbox_on_page:
                             space_to_prev = s["bbox"][1] - prev_span_bbox_on_page[3]
                        
                        span_info = {
                            "text": text,
                            "page_num": page_num + 1,
                            "block_num": block_num,
                            "line_num": line_num,
                            "bbox_x0": s["bbox"][0],
                            "font_size": font_size,
                            "norm_font_size": font_size / doc_common_font_size if doc_common_font_size > 0 else 1,
                            "is_bold": "bold" in s["font"].lower(),
                            "is_italic": "italic" in s["font"].lower(),
                            "y_pos_rel": s["bbox"][1] / page_height if page_height > 0 else 0,
                            "text_len": len(text),
                            "cap_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
                            "toc_match_score": get_toc_match_score(text, page_num + 1, toc),
                            "space_to_prev": space_to_prev,
                        }
                        features.append(span_info)
                        all_span_texts.append(text)
                        prev_span_bbox_on_page = s["bbox"]

    if not features:
        doc.close()
        return pd.DataFrame()

    if EMBEDDING_MODEL:
        print(f"Generating embeddings for {len(all_span_texts)} spans...")
        embeddings = EMBEDDING_MODEL.encode(all_span_texts, show_progress_bar=True)
        
        for i, feat in enumerate(features):
            for j, val in enumerate(embeddings[i]):
                feat[f"emb_{j}"] = val
    
    doc.close()
    return pd.DataFrame(features)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Starting feature extraction for: {pdf_path}")
        df = extract_features(pdf_path)
        if not df.empty:
            print(f"Successfully extracted {len(df)} spans.")
            print("Columns:", df.columns.tolist())
            print("First 5 rows:")
            print(df.head())
            output_csv_path = "parser_test_output.csv"
            df.to_csv(output_csv_path, index=False)
            print(f"Test output saved to {output_csv_path}")
    else:
        print("Usage: python src/parser.py <path_to_pdf>")
