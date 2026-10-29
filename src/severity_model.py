import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class SeverityClassifier:
    """
    ML Severity Classifier module for Civic Grievance Triage.
    Primary Pipeline: TF-IDF + Naive Bayes.
    Comparison Models: Logistic Regression, Decision Tree, KNN.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        self.primary_model = MultinomialNB()
        self.models = {
            "Naive Bayes": self.primary_model,
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5)
        }
        self.trained_models = {}
        self.metrics_report = {}
        self.classes = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        self.is_fitted = False

    def fit_and_evaluate(self, X_text: pd.Series, y_labels: pd.Series, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Fits TF-IDF vectorizer and trains primary and comparison models.
        Evaluates models and returns comprehensive performance report.
        
        Args:
            X_text (pd.Series): Text feature column.
            y_labels (pd.Series): Target severity labels.
            test_size (float): Train/test split fraction.
            random_state (int): Random seed.
            
        Returns:
            dict: Performance metrics comparison dictionary.
        """
        # Train/Test Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_text, y_labels, test_size=test_size, random_state=random_state, stratify=y_labels
        )
        
        # Fit TF-IDF Vectorizer
        X_train_tfidf = self.vectorizer.fit_transform(X_train_raw)
        X_test_tfidf = self.vectorizer.transform(X_test_raw)
        
        report = {}
        
        for name, model in self.models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
            rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            cm = confusion_matrix(y_test, y_pred, labels=self.classes).tolist()
            
            report[name] = {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "confusion_matrix": cm
            }
            self.trained_models[name] = model

        self.metrics_report = report
        self.is_fitted = True
        return report

    def predict_severity(self, text_list: list) -> list:
        """
        Predicts severity using primary Naive Bayes model.
        
        Args:
            text_list (list): List of complaint text strings.
            
        Returns:
            list: List of predicted severity strings.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call fit_and_evaluate() first.")
            
        X_tfidf = self.vectorizer.transform(text_list)
        preds = self.primary_model.predict(X_tfidf)
        return list(preds)

    def predict_proba(self, text_list: list) -> list:
        """
        Returns class probability distribution for text inputs.
        
        Args:
            text_list (list): List of complaint text strings.
            
        Returns:
            list: List of probability dictionaries per class.
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet.")
            
        X_tfidf = self.vectorizer.transform(text_list)
        probs = self.primary_model.predict_proba(X_tfidf)
        classes = list(self.primary_model.classes_)
        
        result = []
        for p in probs:
            result.append({cls: float(prob) for cls, prob in zip(classes, p)})
        return result

    def save_model(self, models_dir: str):
        """
        Serializes vectorizer, primary model, and evaluation report.
        
        Args:
            models_dir (str): Destination path for model artifacts.
        """
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
        joblib.dump(self.primary_model, os.path.join(models_dir, "nb_severity_model.joblib"))
        
        # Save comparison models as well
        for name, m in self.trained_models.items():
            sanitized_name = name.lower().replace(" ", "_")
            joblib.dump(m, os.path.join(models_dir, f"{sanitized_name}_severity_model.joblib"))
            
        joblib.dump(self.metrics_report, os.path.join(models_dir, "severity_metrics_report.joblib"))
        print(f"Saved trained severity models and vectorizer to {models_dir}")

    def load_model(self, models_dir: str):
        """
        Loads serialized vectorizer and primary Naive Bayes model.
        
        Args:
            models_dir (str): Path containing model artifacts.
        """
        vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
        model_path = os.path.join(models_dir, "nb_severity_model.joblib")
        
        if not os.path.exists(vec_path) or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model files not found in {models_dir}")
            
        self.vectorizer = joblib.load(vec_path)
        self.primary_model = joblib.load(model_path)
        self.models["Naive Bayes"] = self.primary_model
        self.trained_models["Naive Bayes"] = self.primary_model
        
        metrics_path = os.path.join(models_dir, "severity_metrics_report.joblib")
        if os.path.exists(metrics_path):
            self.metrics_report = joblib.load(metrics_path)
            
        self.is_fitted = True
        print(f"Successfully loaded severity model from {models_dir}")
