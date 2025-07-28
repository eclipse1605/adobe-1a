import pandas as pd
import joblib
import sys
import os
import json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.parser import extract_features
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = 'models/heading_detection_model.joblib'

def load_model_and_artifacts(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        return None, None, None
    
    try:
        model_payload = joblib.load(model_path)
        model = model_payload['model']
        features = model_payload['features']
        label_encoder = model_payload['label_encoder']
        return model, features, label_encoder
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return None, None, None

def predict_headings(pdf_path: str, model, features: list, label_encoder: LabelEncoder) -> pd.DataFrame:

    df = extract_features(pdf_path)
    if df.empty:
        return pd.DataFrame()

    for feature in features:
        if feature not in df.columns:
            df[feature] = 0
            
    X_to_predict = df[features]
    
    try:
        predictions_encoded = model.predict(X_to_predict)
        df['label'] = label_encoder.inverse_transform(predictions_encoded)
    except Exception as e:
        print(f"Error during prediction or decoding: {e}", file=sys.stderr)
        return pd.DataFrame()

    df = apply_heuristics(df)
    
    headings_df = df[df['label'].isin(['title', 'H1', 'H2', 'H3', 'TOC'])].copy()
    
    return headings_df[['text', 'label', 'page_num', 'block_num', 'line_num']]

def apply_heuristics(df: pd.DataFrame) -> pd.DataFrame:
    
    if 'font_size' not in df.columns or 'is_bold' not in df.columns or 'is_centered' not in df.columns:
        return df
        
    for i, row in df.iterrows():
        if row['font_size'] > 15 and row['is_bold']:
            if row['page_num'] < 2 and row['is_centered'] and row['label'] not in ['title', 'H1']:
                 df.loc[i, 'label'] = 'title'
            elif row['page_num'] < 2 and row['label'] not in ['title', 'H1']:
                 df.loc[i, 'label'] = 'title' if row['page_num'] == 0 else 'H1'
            elif row['label'] not in ['title', 'H1']:
                df.loc[i, 'label'] = 'H1'

    for i, row in df.iterrows():
        if row['starts_with_number'] and not row['keyword_in_text']:
            if row['label'] not in ['H1', 'H2', 'H3']:
                 df.loc[i, 'label'] = 'H2'
        elif row['keyword_in_text'] and row['label'] not in ['title', 'H1']:
             df.loc[i, 'label'] = 'H1'
             
    df.sort_values(by=['page_num', 'block_num', 'line_num'], inplace=True)
    
    new_labels = df['label'].copy()
    last_h1_index = -1
    last_h2_index = -1

    for i in df.index:
        label = df.loc[i, 'label']
        if label == 'H1':
            last_h1_index = i
            last_h2_index = -1
        elif label == 'H2':
            if last_h1_index == -1:
                new_labels.loc[i] = 'H1'
                last_h1_index = i
            else:
                last_h2_index = i
        elif label == 'H3':
            if last_h2_index == -1:
                if last_h1_index != -1:
                    new_labels.loc[i] = 'H2'
                    last_h2_index = i
                else:
                    new_labels.loc[i] = 'H1'
                    last_h1_index = i
    
    df['label'] = new_labels
                
    toc_keywords = ['contents', 'table of contents', 'index']
    for i, row in df.iterrows():
        if any(keyword in row['text'].lower() for keyword in toc_keywords):
            df.loc[i, 'label'] = 'TOC'
            
    return df

def create_json_outline(headings_df: pd.DataFrame, pdf_filename: str) -> str:
    title_df = headings_df[headings_df['label'] == 'title']
    if not title_df.empty:
        best_title = title_df.sort_values(by=['page_num', 'block_num', 'line_num']).iloc[0]
        title = best_title['text']
    else:
        title = os.path.basename(pdf_filename).replace('.pdf', '')

    toc_df = headings_df[headings_df['label'] == 'TOC']
    outline_df = headings_df[headings_df['label'].isin(['H1', 'H2', 'H3'])].copy()
    outline_df.sort_values(by=['page_num', 'block_num', 'line_num'], inplace=True)

    outline = []
    
    if not toc_df.empty:
        toc_df.sort_values(by=['page_num', 'block_num', 'line_num'], inplace=True)
        for i, row in toc_df.iterrows():
            outline.append({
                "level": "TOC",
                "text": row['text'],
                "page": int(row['page_num'])
            })
            
    i = 0
    while i < len(outline_df):
        current_heading = outline_df.iloc[i]
        
        merged_text = current_heading['text']
        
        j = i + 1
        while j < len(outline_df):
            next_heading = outline_df.iloc[j]
            
            if (current_heading['page_num'] == next_heading['page_num'] and
                current_heading['label'] == next_heading['label'] and
                current_heading['block_num'] == next_heading['block_num'] and
                outline_df.iloc[j-1]['line_num'] + 1 == next_heading['line_num']):
                
                merged_text += " " + next_heading['text']
                j += 1
            else:
                break
        
        outline.append({
            "level": current_heading['label'],
            "text": merged_text,
            "page": int(current_heading['page_num'])
        })
        i = j

    json_output = {
        "title": title,
        "outline": outline
    }
    
    return json.dumps(json_output, indent=2)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/detector.py <path_to_pdf_or_directory>")
        sys.exit(1)
        
    path_arg = sys.argv[1]

    if not os.path.exists(path_arg):
        print(f"Error: The path '{path_arg}' was not found.")
        sys.exit(1)

    pdf_files = []
    if os.path.isdir(path_arg):
        pdf_files = sorted([os.path.join(path_arg, f) for f in os.listdir(path_arg) if f.lower().endswith('.pdf')])
    elif os.path.isfile(path_arg) and path_arg.lower().endswith('.pdf'):
        pdf_files.append(path_arg)
    
    if not pdf_files:
        print("No PDF files found to process.")
        sys.exit(0)
    
    model, model_features, le = load_model_and_artifacts()

    if not all([model, model_features, le]):
        sys.exit(1)

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        print(f"\n--- Processing: {pdf_name} ---")
        
        predicted_headings = predict_headings(pdf_path, model, model_features, le)
        
        if not predicted_headings.empty:
            json_output = create_json_outline(predicted_headings, pdf_name)
            print(json_output)
        else:
            print(json.dumps({"title": pdf_name, "outline": []}, indent=2))
