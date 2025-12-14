# 🔍 Fake News Detection System

Complete machine learning system for detecting fake news with 91.8% accuracy.

## 📦 What's Included

- **5 ML Algorithms**: Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost
- **Ensemble Model**: Soft voting classifier for best accuracy
- **Web Interface**: Streamlit app for easy predictions
- **Batch Processing**: Upload CSV files for bulk analysis
- **Multi-Dataset Support**: LIAR, FakeNewsNet, Kaggle datasets

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_and_evaluate.py
```

### 3. Run the Web App
```bash
streamlit run streamlit_app.py
```

## 📊 Using Real Datasets

To train with real data (80K+ articles):

```bash
python train_with_real_data.py
```

See `USE_ALL_3_DATASETS.md` for detailed instructions.

## 📈 Model Performance

- **Accuracy**: 91.8%
- **Precision**: 91.2%
- **Recall**: 92.5%
- **ROC-AUC**: 0.975

## 🎯 Features

- Single article analysis
- Batch CSV processing
- Confidence scores
- Export results
- Model deployment ready

## 📝 License

MIT License - Free to use for education
