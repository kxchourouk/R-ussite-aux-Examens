# 🎓 Réussite aux Examens - Student Success Prediction System

A Bayesian machine learning project that predicts student exam success based on three key factors: revision, confidence, and health status.

## 📋 Overview

This project implements a complete machine learning pipeline including:
- Synthetic data generation or real dataset loading
- Bayesian probabilistic model training
- Model validation and performance metrics
- Sensitivity analysis of influencing factors
- Interactive web interface for predictions

## 🎯 Features

- **Bayesian Classification**: Predicts exam success probability using conditional probabilities
- **Factor Analysis**: Identifies which factors (revision, confidence, health) have the most impact
- **Model Validation**: Comprehensive metrics (accuracy, precision, recall, F1-score, confusion matrix)
- **Interactive Web UI**: User-friendly interface for students to get personalized predictions
- **Visualization**: Charts and graphs for sensitivity analysis and model performance

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this project**

2. **Create a virtual environment:**
```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# OR activate it (Windows Command Prompt)
venv\Scripts\activate.bat
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

> **Note**: Always activate the virtual environment before running the project scripts.

## 🚀 Usage

### Option 1: Full Pipeline (Recommended)
Run the complete analysis with all features:

```bash
python lancer_projet_complet.py
```

This executes:
1. Data generation (`generate_data.py`)
2. Model training (`train_bayesian_model_simple.py`)
3. Sensitivity analysis (`analyse_sensibilite.py`)
4. Model validation (`validation_modele.py`)
5. Instructions for launching the web interface

### Option 2: Quick Start
Run only the essential components:

```bash
python lancer_tout.py
```

This executes:
1. Data generation
2. Model training

### Option 3: Individual Scripts
Run components separately:

```bash
# Generate synthetic data
python generate_data.py

# OR use real Kaggle dataset
python train_model.py

# Train the Bayesian model
python train_bayesian_model_simple.py

# Analyze factor sensitivity
python analyse_sensibilite.py

# Validate model performance
python validation_modele.py
```

## 🌐 Web Interface

After running the pipeline, launch the web interface:

### Method 1: Direct Opening
- Double-click `index.html` in your file explorer

### Method 2: Local Web Server (Recommended)
```bash
python -m http.server 8000
```
Then open your browser to: `http://localhost:8000`

## 📊 Project Structure

```
Réussite aux Examens/
├── generate_data.py                    # Generates synthetic student data
├── train_model.py                      # Loads real Kaggle dataset (alternative)
├── train_bayesian_model_simple.py      # Trains Bayesian classifier
├── validation_modele.py                # Validates model performance
├── analyse_sensibilite.py              # Analyzes factor importance
├── lancer_tout.py                      # Quick launch script
├── lancer_projet_complet.py            # Full pipeline launcher
├── index.html                          # Interactive web interface
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── Output Files:
    ├── ma_base_donnees.csv             # Training dataset
    ├── bayesian_model_simple.joblib    # Trained model (Python)
    ├── model_probabilities.json        # Predictions (JSON)
    ├── analyse_sensibilite.json        # Sensitivity results
    ├── analyse_sensibilite.png         # Sensitivity charts
    ├── validation_resultats.json       # Validation metrics
    └── validation_modele.png           # Performance visualizations
```

## 🔍 How It Works

### Data Model
The system uses 4 binary variables:
- **Revision** (0/1): Whether the student studied
- **Confiance** (0/1): Student's confidence level
- **Malade** (0/1): Health status (sick or healthy)
- **Succes** (0/1): Exam outcome (pass or fail)

### Bayesian Model
The model calculates: **P(Success | Revision, Confidence, Health)**

Using Bayes' theorem and conditional probabilities learned from the training data, it predicts success probability for all 8 possible input combinations (2³).

### Example Predictions
- **High Success**: Revision=Yes, Confidence=Yes, Health=Good → ~85% success
- **Low Success**: Revision=No, Confidence=No, Health=Sick → ~15% success

## 📈 Output Files Explained

| File | Description |
|------|-------------|
| `ma_base_donnees.csv` | Training data (1000 students) |
| `bayesian_model_simple.joblib` | Serialized Python model |
| `model_probabilities.json` | Pre-computed predictions for web interface |
| `analyse_sensibilite.json` | Factor impact analysis results |
| `analyse_sensibilite.png` | Visual charts of factor importance |
| `validation_resultats.json` | Performance metrics (accuracy, F1, etc.) |
| `validation_modele.png` | Confusion matrix & ROC curve |

## 📊 Performance Metrics

The validation script provides:
- **Accuracy**: Overall prediction correctness
- **Precision**: Reliability of positive predictions
- **Recall**: Detection rate of actual successes
- **F1-Score**: Balanced precision-recall metric
- **Confusion Matrix**: True/false positives and negatives
- **ROC Curve**: Threshold sensitivity analysis

## 🎨 Sensitivity Analysis

Identifies the relative impact of each factor:
1. **Revision**: Typically +25-35 percentage points
2. **Confidence**: Typically +15-25 percentage points
3. **Health**: Typically +10-20 percentage points

## 🛠️ Customization

### Modify Data Generation
Edit `generate_data.py` to adjust:
- Number of students (`n_students`)
- Base success probability
- Factor weight multipliers
- Random seed for reproducibility

### Adjust Model Parameters
Edit `train_bayesian_model_simple.py` to:
- Change probability calculation method
- Adjust smoothing parameters
- Modify prediction thresholds

## 📝 License

This is an educational project for demonstration purposes.

## 🤝 Contributing

This project is designed for learning Bayesian machine learning concepts. Feel free to fork and enhance!

## 📧 Support

For questions or issues, please review the code comments in each script for detailed explanations.

---

**Made with ❤️ for student success prediction**
