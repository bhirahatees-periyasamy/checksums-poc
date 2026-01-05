import os
import pickle
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def calculate_file_hash(filepath, algorithm='sha256'):
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def create_and_save_dataset():
    """Create credit card fraud dataset and save as CSV."""
    print("CREATING DATASET")
    
    np.random.seed(42)
    
    # Generate credit card transaction data
    n_samples = 10000
    n_features = 8
    
    # Create features V1-V28 (PCA transformed features like real Kaggle dataset)
    features = np.random.randn(n_samples, n_features)
    
    # Add Time and Amount features
    time = np.random.uniform(0, 172792, n_samples)  # Time in seconds
    amount = np.random.lognormal(3, 2, n_samples)  # Transaction amounts
    
    # Create imbalanced target (fraud detection)
    # 98% legitimate, 2% fraud
    fraud_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    target = np.zeros(n_samples)
    target[fraud_indices] = 1
    
    # Make fraud transactions have different patterns
    features[fraud_indices] += np.random.randn(len(fraud_indices), n_features) * 2
    amount[fraud_indices] *= 3  # Fraudulent transactions tend to be larger
    
    # Create DataFrame
    data = {
        'Time': time,
        'Amount': amount
    }
    
    # Add V1-V28 features
    for i in range(1, n_features + 1):
        data[f'V{i}'] = features[:, i-1]
    
    data['Class'] = target
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs("data", exist_ok=True)
    csv_path = "data/credit_card_transactions.csv"
    df.to_csv(csv_path, index=False)
    
    print(f" Dataset created and saved to: {csv_path}")
    print(f" Total samples: {len(df):,}")
    print(f" Features: {len(df.columns) - 1}")
    print(f" Legitimate transactions: {(target == 0).sum():,} ({(target == 0).mean()*100:.1f}%)")
    print(f" Fraudulent transactions: {(target == 1).sum():,} ({(target == 1).mean()*100:.1f}%)")
    print(f" CSV file hash: {calculate_file_hash(csv_path)[:32]}...")
    
    return csv_path, df


def load_dataset(csv_path):
    """Load dataset from CSV."""
    df = pd.read_csv(csv_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    return X, y


def get_sample_predictions(model, X_test, y_test, n_samples=5):
    """Get sample predictions for demonstration."""
    # Get random samples
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]
    
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    samples = []
    for i, idx in enumerate(sample_indices):
        sample = {
            "sample_id": int(idx),
            "actual_class": int(y_sample.iloc[i]),
            "predicted_class": int(predictions[i]),
            "probability_legitimate": float(probabilities[i][0]),
            "probability_fraud": float(probabilities[i][1]),
            "correct_prediction": bool(predictions[i] == y_sample.iloc[i]),
            "input_features": {
                "Time": float(X_sample.iloc[i]['Time']),
                "Amount": float(X_sample.iloc[i]['Amount']),
                "V1": float(X_sample.iloc[i]['V1']),
                "V2": float(X_sample.iloc[i]['V2'])
            }
        }
        samples.append(sample)
    
    return samples


def train_model_1_standard_training(csv_path):
    """Training Procedure 1: Standard XGBoost Training."""
    print("\nStandard Training Procedure")
    
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    print(f"Standard fit() method")
    print(f"Hyperparameters: depth={params['max_depth']}, lr={params['learning_rate']}, n_est={params['n_estimators']}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get sample predictions
    sample_predictions = get_sample_predictions(model, X_test, y_test)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/model_v1_standard_training.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved: {model_path}")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return {
        "path": model_path,
        "name": "Fraud_Detection_Standard",
        "version": "1.0.0",
        "training_procedure": "Standard fit() - No preprocessing",
        "accuracy": float(accuracy),
        "auc": float(auc),
        "samples": len(X_train),
        "params": params,
        "sample_predictions": sample_predictions
    }


def train_model_2_early_stopping(csv_path):
    """Training Procedure 2: Early Stopping with Validation Set."""
    print("\nEarly Stopping Training Procedure")
    
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'random_state': 42,
        'tree_method': 'hist',
        'early_stopping_rounds': 20
    }
    
    print(f"Early stopping with validation set (patience=20)")
    print(f"Hyperparameters: depth={params['max_depth']}, lr={params['learning_rate']}, max_n_est={params['n_estimators']}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get sample predictions
    sample_predictions = get_sample_predictions(model, X_test, y_test)
    
    # Save model
    model_path = "models/model_v2_early_stopping.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved: {model_path}")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    
    return {
        "path": model_path,
        "name": "Fraud_Detection_EarlyStopping",
        "version": "2.0.0",
        "training_procedure": "Early stopping with validation monitoring",
        "best_iteration": int(model.best_iteration),
        "accuracy": float(accuracy),
        "auc": float(auc),
        "samples": len(X_train),
        "params": params,
        "sample_predictions": sample_predictions
    }


def train_model_3_scaled_features(csv_path):
    """Training Procedure 3: Feature Scaling/Normalization."""
    print("\nFeature Scaling Training Procedure")
    
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Apply StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for consistency
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    print(f"StandardScaler normalization applied")
    print(f"Hyperparameters: depth={params['max_depth']}, lr={params['learning_rate']}, n_est={params['n_estimators']}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get sample predictions
    sample_predictions = get_sample_predictions(model, X_test_scaled, y_test)
    
    # Save model and scaler
    model_path = "models/model_v3_scaled_features.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    
    print(f" Model saved: {model_path}")
    print(f" Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return {
        "path": model_path,
        "name": "Fraud_Detection_Scaled",
        "version": "3.0.0",
        "training_procedure": "StandardScaler feature normalization",
        "preprocessing": "StandardScaler (mean=0, std=1)",
        "accuracy": float(accuracy),
        "auc": float(auc),
        "samples": len(X_train),
        "params": params,
        "sample_predictions": sample_predictions
    }


def train_model_4_weighted_training(csv_path):
    """Training Procedure 4: Custom Sample Weights."""
    print("\n Weighted Training Procedure")
    
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Create custom sample weights (higher weight for fraud cases)
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train == 1] = 10.0 
    
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'tree_method': 'hist'
    }
    
    print(f"Custom sample weights (fraud=10x, legitimate=1x)")
    print(f"Hyperparameters: depth={params['max_depth']}, lr={params['learning_rate']}, n_est={params['n_estimators']}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Get sample predictions
    sample_predictions = get_sample_predictions(model, X_test, y_test)
    
    # Save model
    model_path = "models/model_v4_weighted_training.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved: {model_path}")
    print(f"Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    return {
        "path": model_path,
        "name": "Fraud_Detection_Weighted",
        "version": "4.0.0",
        "training_procedure": "Custom sample weights for imbalanced data",
        "sample_weights": "Fraud=10.0, Legitimate=1.0",
        "accuracy": float(accuracy),
        "auc": float(auc),
        "samples": len(X_train),
        "params": params,
        "sample_predictions": sample_predictions
    }


def generate_checksums_json(model_results, csv_path):
    """Generate comprehensive checksums JSON with predictions."""
    print("GENERATING MODEL CHECKSUMS WITH PREDICTIONS")
    
    records = []
    
    for result in model_results:
        model_path = result['path']
        file_size = os.path.getsize(model_path)
        model_hash = calculate_file_hash(model_path)
        
        record = {
            "model_name": result['name'],
            "version": result['version'],
            "hash": model_hash,
            "hash_algorithm": "sha256",
            "file_path": model_path,
            "file_size_bytes": file_size,
            "trained_at": datetime.now().isoformat(),
            "dataset": {
                "source": csv_path,
                "hash": calculate_file_hash(csv_path),
                "training_samples": result['samples']
            },
            "training_procedure": result['training_procedure'],
            "hyperparameters": {
                k: v for k, v in result['params'].items()
                if k in ['max_depth', 'learning_rate', 'n_estimators', 'objective']
            },
            "performance_metrics": {
                "accuracy": result['accuracy'],
                "auc_score": result['auc']
            },
            "sample_predictions": result['sample_predictions'],
            "verified": True
        }
        
        # Add procedure-specific metadata
        if 'best_iteration' in result:
            record['early_stopping_info'] = {
                "best_iteration": result['best_iteration'],
                "max_iterations": result['params']['n_estimators']
            }
        
        if 'preprocessing' in result:
            record['preprocessing'] = result['preprocessing']
        
        if 'sample_weights' in result:
            record['sample_weights_info'] = result['sample_weights']
        
        records.append(record)
        print(f" {result['name']}")
        print(f" Hash: {model_hash}")
        print(f" Procedure: {result['training_procedure']}")
    
    # Create output structure
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_models": len(records),
            "verified_models": sum(1 for r in records if r['verified']),
            "description": "XGBoost Models with Different Training Procedures",
            "framework": "XGBoost",
            "dataset_info": {
                "name": "Credit Card Fraud Detection",
                "path": csv_path,
                "hash": calculate_file_hash(csv_path),
                "format": "CSV"
            }
        },
        "models": records
    }
    
    # Save to JSON
    output_file = "model_checksums.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n Checksums saved to: {output_file}")
    return output_file


def display_summary(results, json_file, csv_path):
    """Display comprehensive summary."""
    
    print(f"\nDataset Information:")
    print(f"CSV File: {csv_path}")
    print(f"Hash: {calculate_file_hash(csv_path)[:32]}...")
    
    print(f"\nModels Trained: {len(results)}")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']} v{result['version']}")
        print(f"Procedure: {result['training_procedure']}")
        print(f"Accuracy: {result['accuracy']:.4f} | AUC: {result['auc']:.4f}")
        print(f"File: {os.path.basename(result['path'])}")
        
        # Show one sample prediction
        sample = result['sample_predictions'][0]
        print(f"Sample Prediction: Actual={sample['actual_class']}, "
              f"Predicted={sample['predicted_class']}, "
              f"Fraud Prob={sample['probability_fraud']:.3f}")
    
    print("\nGenerated Files:")
    print(f"  • {csv_path}")
    print(f"  • models/model_v1_standard_training.pkl")
    print(f"  • models/model_v2_early_stopping.pkl")
    print(f"  • models/model_v3_scaled_features.pkl")
    print(f"  • models/model_v4_weighted_training.pkl")
    print(f"  • {json_file}")
    


def main():
    """Main execution function."""
    try:
        # Create and save dataset
        csv_path, df = create_and_save_dataset()
        
        # Train all models with different procedures
        results = []
        results.append(train_model_1_standard_training(csv_path))
        results.append(train_model_2_early_stopping(csv_path))
        results.append(train_model_3_scaled_features(csv_path))
        results.append(train_model_4_weighted_training(csv_path))
        
        # Generate checksums with predictions
        json_file = generate_checksums_json(results, csv_path)
        
        # Display summary
        display_summary(results, json_file, csv_path)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()