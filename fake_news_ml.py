"""
FAKE NEWS DETECTION SYSTEM - ML PIPELINE (Fixed for Ensemble)
Complete machine learning system with multiple algorithms and ensemble voting
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import pickle
import os

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Using 4 algorithms instead of 5.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DataPreprocessor:
    """Preprocesses and prepares data for ML models"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    def preprocess(self, texts):
        """Convert texts to TF-IDF vectors"""
        texts = [str(t).lower().strip() for t in texts]
        X = self.vectorizer.fit_transform(texts)
        return X
    
    def transform(self, texts):
        """Transform new texts using fitted vectorizer"""
        texts = [str(t).lower().strip() for t in texts]
        return self.vectorizer.transform(texts)


class FakeNewsDetectionModel:
    """Main fake news detection model"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.ensemble_model = None
        self.vectorizer = None
        
    def prepare_data(self, texts, labels):
        """Prepare data for training"""
        X = self.preprocessor.preprocess(texts)
        self.vectorizer = self.preprocessor.vectorizer
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression"""
        print("  Training Logistic Regression...")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        acc = model.score(X_test, y_test)
        print(f"    ✓ Accuracy: {acc:.4f}")
    
    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        """Train Naive Bayes"""
        print("  Training Naive Bayes...")
        model = MultinomialNB()
        model.fit(X_train, y_train)
        self.models['naive_bayes'] = model
        acc = model.score(X_test, y_test)
        print(f"    ✓ Accuracy: {acc:.4f}")
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM with probability=True for ensemble"""
        print("  Training SVM...")
        # Use SVC with probability=True for predict_proba
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)
        self.models['svm'] = model
        acc = model.score(X_test, y_test)
        print(f"    ✓ Accuracy: {acc:.4f}")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest"""
        print("  Training Random Forest...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        acc = model.score(X_test, y_test)
        print(f"    ✓ Accuracy: {acc:.4f}")
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost"""
        if not HAS_XGBOOST:
            print("  Skipping XGBoost (not installed)...")
            return
        
        print("  Training XGBoost...")
        model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        acc = model.score(X_test, y_test)
        print(f"    ✓ Accuracy: {acc:.4f}")
    
    def create_ensemble(self):
        """Create ensemble voting classifier"""
        estimators = [
            ('lr', self.models['logistic_regression']),
            ('nb', self.models['naive_bayes']),
            ('svm', self.models['svm']),
            ('rf', self.models['random_forest']),
        ]
        
        if 'xgboost' in self.models:
            estimators.append(('xgb', self.models['xgboost']))
        
        # Use soft voting for probability estimates
        self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
            }
        
        if self.ensemble_model:
            y_pred = self.ensemble_model.predict(X_test)
            try:
                y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                roc_auc = 0.0
            
            results['ensemble'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc,
            }
        
        return results
    
    def predict(self, text):
        """Predict if text is fake news"""
        if self.ensemble_model is None:
            return None
        
        X = self.vectorizer.transform([text])
        pred = self.ensemble_model.predict(X)[0]
        
        try:
            confidence = max(self.ensemble_model.predict_proba(X)[0])
        except:
            confidence = 0.5
        
        return int(pred), float(confidence)
    
    def save_models(self):
        """Save all models to disk"""
        os.makedirs('./models', exist_ok=True)
        
        for name, model in self.models.items():
            with open(f'./models/{name}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        if self.ensemble_model:
            with open('./models/ensemble.pkl', 'wb') as f:
                pickle.dump(self.ensemble_model, f)
        
        if self.vectorizer:
            with open('./models/vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        print("✓ Models saved to ./models/")
    
    def load_models(self):
        """Load models from disk"""
        try:
            for name in ['logistic_regression', 'naive_bayes', 'svm', 'random_forest']:
                with open(f'./models/{name}.pkl', 'rb') as f:
                    self.models[name] = pickle.load(f)
            
            try:
                with open('./models/xgboost.pkl', 'rb') as f:
                    self.models['xgboost'] = pickle.load(f)
            except FileNotFoundError:
                pass
            
            with open('./models/ensemble.pkl', 'rb') as f:
                self.ensemble_model = pickle.load(f)
            
            with open('./models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
