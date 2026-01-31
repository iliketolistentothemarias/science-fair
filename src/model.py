import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from .utils import setup_logging

logger = setup_logging()

def load_and_merge_data(radiomics_csv, deep_csv, clinical_csv):
    """
    Merges radiomics, deep features, and clinical labels.
    """
    try:
        df_rad = pd.read_csv(radiomics_csv)
        df_deep = pd.read_csv(deep_csv)
        df_clin = pd.read_csv(clinical_csv) # Must have PatientID and EGFR_Label (0/1)
        
        # Merge on PatientID
        df = df_rad.merge(df_deep, on='PatientID', how='inner')
        df = df.merge(df_clin, on='PatientID', how='inner')
        
        logger.info(f"Merged dataframe shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading/merging data: {e}")
        return None

def train_evaluate_pipeline(df, label_col='EGFR_Label', n_folds=5):
    """
    Runs the full pipeline: Scaling -> LASSO -> XGBoost with CV.
    """
    X = df.drop(columns=['PatientID', label_col])
    # Drop non-numeric if any
    X = X.select_dtypes(include=[np.number])
    y = df[label_col]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    auc_scores = []
    acc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 2. Feature Selection (LASSO)
        # lassoCV automatically picks best alpha
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000).fit(X_train_scaled, y_train)
        model_sel = SelectFromModel(lasso, prefit=True)
        X_train_sel = model_sel.transform(X_train_scaled)
        X_val_sel = model_sel.transform(X_val_scaled)
        
        n_features = X_train_sel.shape[1]
        logger.info(f"Fold {fold+1}: Selected {n_features} features")
        
        if n_features == 0:
            logger.warning(f"Fold {fold+1}: No features selected by LASSO. Skipping.")
            continue
        
        # 3. XGBoost
        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        
        clf.fit(X_train_sel, y_train)
        
        # Predict
        y_pred_prob = clf.predict_proba(X_val_sel)[:, 1]
        y_pred = clf.predict(X_val_sel)
        
        auc = roc_auc_score(y_val, y_pred_prob)
        acc = accuracy_score(y_val, y_pred)
        
        auc_scores.append(auc)
        acc_scores.append(acc)
        
        logger.info(f"Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}")
        
    mean_auc = np.mean(auc_scores)
    mean_acc = np.mean(acc_scores)
    
    logger.info(f"Mean AUC: {mean_auc:.4f} +/- {np.std(auc_scores):.4f}")
    logger.info(f"Mean Acc: {mean_acc:.4f} +/- {np.std(acc_scores):.4f}")
    
    return mean_auc, mean_acc
