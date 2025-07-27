import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tqdm import tqdm
import optuna

DATA_DIRECTORY = 'data/labeled_data'
MODEL_OUTPUT_PATH = 'models/heading_detection_model.joblib'
ARTIFACTS_DIR = 'models'
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_preprocess_data(directory_path: str):
    all_data = []
    files_to_process = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not files_to_process:
        logging.error("No CSV files found in the directory.")
        return None, None

    logging.info(f"Loading data from {len(files_to_process)} files...")
    for filename in tqdm(files_to_process, desc="Loading data"):
        file_path = os.path.join(directory_path, filename)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
        except Exception as e:
            logging.error(f"Error loading {filename}: {e}")
            continue
            
    if not all_data:
        logging.error("No data was loaded. Exiting.")
        return None, None
        
    combined_df = pd.concat(all_data, ignore_index=True)
    
    combined_df['label'].fillna('body', inplace=True)
    
    label_counts = combined_df['label'].value_counts()
    rare_labels = label_counts[label_counts < 2].index
    if not rare_labels.empty:
        logging.warning(f"Removing rare labels with only 1 instance: {list(rare_labels)}")
        combined_df = combined_df[~combined_df['label'].isin(rare_labels)]
    
    label_encoder = LabelEncoder()
    combined_df['label_encoded'] = label_encoder.fit_transform(combined_df['label'])
    
    return combined_df, label_encoder

def train_and_evaluate(df: pd.DataFrame, label_encoder: LabelEncoder, model_output_path: str):
    features = [col for col in df.columns if col not in ['text', 'label', 'label_encoded']]
    X = df[features]
    y = df['label_encoded']
    class_names = list(label_encoder.classes_)

    def objective(trial):
        param = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'num_class': len(class_names),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1
        }

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      eval_metric='multi_logloss',
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            preds = model.predict(X_val)
            scores.append(accuracy_score(y_val, preds))
        
        return np.mean(scores)

    logging.info("Starting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50) 
    
    logging.info(f"Best trial score: {study.best_value}")
    logging.info("Best parameters found: ", study.best_params)

    best_params = study.best_params
    best_params['objective'] = 'multiclass'
    best_params['metric'] = 'multi_logloss'
    best_params['num_class'] = len(class_names)
    best_params['random_state'] = RANDOM_STATE
    best_params['n_jobs'] = -1
    best_params['class_weight'] = 'balanced'

    logging.info("\n--- Training Final Model on all data with best parameters ---")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X, y)
    
    logging.info("Evaluating final model on a hold-out set for reporting...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    final_model.fit(X_train, y_train) 
    preds = final_model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    logging.info(f"Final Model Validation Accuracy: {accuracy:.4f}")
    logging.info("Final Model Classification Report:\n" + classification_report(y_val, preds, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Final Model Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'validation_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    feature_importances = pd.DataFrame({'feature': features, 'importance': final_model.feature_importances_})
    feature_importances.sort_values('importance', ascending=False, inplace=True)
    
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importances)
    plt.title('Feature Importances from Final Model')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'validation_feature_importance.png'), bbox_inches='tight')
    plt.close()
    
    model_payload = {
        'model': final_model,
        'features': features,
        'label_encoder': label_encoder
    }
    joblib.dump(model_payload, model_output_path)
    logging.info(f"Final model and artifacts saved to {model_output_path}")

if __name__ == '__main__':
    if not os.path.exists(ARTIFACTS_DIR):
        os.makedirs(ARTIFACTS_DIR)
        
    df, le = load_and_preprocess_data(DATA_DIRECTORY)
    if df is not None and le is not None:
        train_and_evaluate(df, le, MODEL_OUTPUT_PATH)
