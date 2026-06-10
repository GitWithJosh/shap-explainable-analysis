# xai-credit-risk

Explainability analysis of a LightGBM credit risk model using SHAP — covering global feature importance, individual prediction breakdowns, and false positive/negative case studies.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-2980B9?style=flat-square&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-FF6B35?style=flat-square&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

## Overview

A LightGBM classifier is trained on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) Kaggle dataset to predict the probability of financial distress. The focus of the project is not the model itself but its interpretability: SHAP values are used to explain predictions at both the population and individual level, with particular attention to understanding why the model makes mistakes. The analysis is available as a self-contained Jupyter notebook and as an interactive Streamlit app. This project was submitted as a seminar paper and graded 1.1.

## Analysis Approach

| Method | What it shows |
|---|---|
| SHAP Feature Importance | Which features drive predictions globally |
| Waterfall Plots | How each feature pushes a single prediction up or down |
| Probability Space Analysis | Model confidence distribution across the dataset |
| Logit Space Analysis | Behaviour near the decision boundary |
| False Positive Case Study | Why low-risk borrowers are sometimes flagged |
| False Negative Case Study | Why high-risk borrowers sometimes pass |

## Quick Start

**Option 1 — Notebook**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jupyter
# Download dataset from https://www.kaggle.com/c/GiveMeSomeCredit/data
# Place cs-training.csv and cs-test.csv into GiveMeSomeCredit/
jupyter notebook notebook/shap_demonstration.ipynb
```

**Option 2 — Streamlit App**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap streamlit plotly
# Place dataset as above
cd app && python start_app.py
```

> The dataset is not included for licensing reasons. A Kaggle account (free) is required to download it.

## Project Structure

```
xai-credit-risk/
├── notebook/
│   ├── shap_demonstration.ipynb   # Main analysis
│   └── figs/                      # Exported figures (SVG + PDF)
│       ├── feature_importance_lightgbm_shap.*
│       ├── false_positive_shap_waterfall.*
│       ├── false_negative_shap_waterfall.*
│       └── confusion_matrix_lightgbm.*
├── app/
│   ├── streamlit_app.py
│   ├── model_utils.py
│   ├── model_components.pkl       # Pre-fitted model + explainer
│   ├── pages/
│   │   ├── shap_analysis.py
│   │   └── documentation.py
│   └── requirements.txt
└── GiveMeSomeCredit/              # Dataset directory (not tracked)
```
