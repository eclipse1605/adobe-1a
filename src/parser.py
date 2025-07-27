import fitz
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import List, Dict, Any
from tqdm import tqdm
import re

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

    features_list = []
    prev_y1 = 0.0

    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width
        blocks = page.get_text("dict")["blocks"]
        
        for block_num, b in enumerate(blocks):
            if b['type'] == 0:  # Text block
                for line_num, l in enumerate(b["lines"]):
                    line_text = "".join([s["text"] for s in l["spans"]]).strip()
                    if not line_text:
                        continue

                    first_span = l["spans"][0]
                    font_size = first_span["size"]
                    
                    current_y0 = l["bbox"][1]
                    space_to_prev = current_y0 - prev_y1 if prev_y1 > 0 else 0.0
                    prev_y1 = l["bbox"][3]
                    
                    line_info = {
                        "text": line_text,
                        "page_num": page_num + 1,
                        "block_num": block_num,
                        "line_num": line_num,
                        "bbox_x0": l["bbox"][0],
                        "bbox_x1": l["bbox"][2],
                        "page_width": page_width,
                        "font_size": font_size,
                        "norm_font_size": font_size / doc_common_font_size if doc_common_font_size > 0 else 1,
                        "is_bold": "bold" in first_span["font"].lower(),
                        "is_italic": "italic" in first_span["font"].lower(),
                        "y_pos_rel": first_span["bbox"][1] / page_height if page_height > 0 else 0,
                        "text_len": len(line_text),
                        "cap_ratio": sum(1 for c in line_text if c.isupper()) / len(line_text) if line_text else 0,
                        "toc_match_score": get_toc_match_score(line_text, page_num + 1, toc),
                        "space_to_prev": space_to_prev
                    }
                    features_list.append(line_info)
        
        prev_y1 = 0.0

    if not features_list:
        doc.close()
        return pd.DataFrame()

    df = pd.DataFrame(features_list)
    df = add_text_based_features(df)
    doc.close()
    return df

def add_text_based_features(df: pd.DataFrame) -> pd.DataFrame:
    df['center_x'] = (df['bbox_x0'] + df['bbox_x1']) / 2
    page_mid_x = df['page_width'] / 2
    tolerance = df['page_width'] * 0.1
    df['is_centered'] = (df['center_x'] > (page_mid_x - tolerance)) & (df['center_x'] < (page_mid_x + tolerance))

    number_pattern = re.compile(r'^\s*(\d+(\.\d+)*|[A-Za-z])\.\s+|^\s*\(\s*(\d+|[ivx]+|[A-Za-z])\s*\)\s*')
    df['starts_with_number'] = df['text'].apply(lambda x: bool(number_pattern.match(x)))

    keywords = ['abstract', 'introduction', 'conclusion', 'summary', 'references', 'appendix', 'chapter', 'section', 'acknowledgements', 'contents']
    keyword_pattern = re.compile(r'\b(' + '|'.join(keywords) + r')\b', re.IGNORECASE)
    df['keyword_in_text'] = df['text'].apply(lambda x: bool(keyword_pattern.search(x)))
    
    df = df.drop(columns=['center_x', 'page_width', 'bbox_x0', 'bbox_x1'])
    
    return df

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"Starting feature extraction for: {pdf_path}")
        df = extract_features(pdf_path)
        if not df.empty:
            print(f"Successfully extracted {len(df)} lines.")
            print("Columns:", df.columns.tolist())
            print("First 5 rows:")
            print(df.head())
            output_csv_path = "parser_test_output.csv"
            df.to_csv(output_csv_path, index=False)
            print(f"Test output saved to {output_csv_path}")
    else:
        print("Usage: python src/parser.py <path_to_pdf>")
