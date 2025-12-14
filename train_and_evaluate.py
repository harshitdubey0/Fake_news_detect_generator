"""
COMPLETE TRAINING AND EVALUATION PIPELINE
Trains 5 algorithms with synthetic data
"""

from fake_news_ml import FakeNewsDetectionModel
import random

def generate_sample_data():
    """Generate sample fake news data"""
    real_news = [
        "Government announces new education policy",
        "Scientists discover new species in Amazon",
        "Company reports quarterly earnings",
    ] * 100

    fake_news = [
        "Miracle cure discovered doctors don't want you to know",
        "Celebrity scandal shocking revelation",
        "Government conspiracy uncovered",
    ] * 100

    texts = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)

    return texts, labels

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FAKE NEWS DETECTION - TRAINING PIPELINE")
    print("="*70)

    texts, labels = generate_sample_data()
    print(f"\nLoaded {len(texts)} articles")

    model = FakeNewsDetectionModel()
    X_train, X_test, y_train, y_test = model.prepare_data(texts, labels)

    print("\nTraining Models...")
    model.train_logistic_regression(X_train, y_train, X_test, y_test)
    model.train_naive_bayes(X_train, y_train, X_test, y_test)
    model.train_svm(X_train, y_train, X_test, y_test)
    model.train_random_forest(X_train, y_train, X_test, y_test)
    model.train_xgboost(X_train, y_train, X_test, y_test)

    print("\nCreating Ensemble...")
    model.create_ensemble()
    model.ensemble_model.fit(X_train, y_train)

    print("\nEvaluating...")
    results = model.evaluate_all_models(X_test, y_test)

    print(f"\nEnsemble Accuracy: {results['ensemble']['accuracy']:.4f}")
    model.save_models()
    print("\n✓ Training Complete!")
