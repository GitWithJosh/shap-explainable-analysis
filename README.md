# SHAP-basierte Explainable AI Analyse

## Implementierung für Kreditrisikobewertung

Dieses Repository enthält eine umfassende Analyse zur Erklärung von Machine Learning-Vorhersagen im Kontext der Kreditrisikobewertung mit SHAP (SHapley Additive exPlanations).

### 📊 Datensatz

> ⚠️ **Hinweis:** Der Datensatz ist aus Lizenzgründen nicht im Repository enthalten. Sie müssen ihn separat von Kaggle herunterladen (siehe Quick Start Anweisungen).

**[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/overview)** - Kaggle Competition Datensatz

### 🎯 Methodik

Diese Analyse verwendet **SHAP (SHapley Additive exPlanations)** zur Erklärung von Machine Learning-Vorhersagen. Das implementierte **LightGBM-Modell** wird durch verschiedene XAI-Techniken interpretierbar gemacht:

- **Feature Importance Analyse** mit SHAP Values
- **Waterfall Plots** für individuelle Vorhersageerklärungen
- **False Positive/Negative Analyse** zur Modellvalidierung
- **Wahrscheinlichkeits- und Logit-Raum Visualisierungen**

### 📝 Quick Start Anweisungen

Wählen Sie eine der folgenden Optionen für den Einstieg:

#### **Option 1: Jupyter Notebook**

1. **Abhängigkeiten installieren:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jupyter
   ```
2. **Datensatz herunterladen:**
   - Besuchen Sie die [Give Me Some Credit Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data)
   - Melden Sie sich bei Kaggle an (kostenlos)
   - Laden Sie die Dateien `cs-training.csv`, `cs-test.csv`, `Data Dictionary.xls`, `sampleEntry.csv` herunter
   - Erstellen Sie einen Ordner `GiveMeSomeCredit/` im Projektverzeichnis und legen Sie die Dateien dort ab
3. **Jupyter Notebook starten:**
   ```bash
   cd notebook
   jupyter notebook shap_demonstration.ipynb
   ```

#### **Option 2: Streamlit Web-App**

1. **Abhängigkeiten installieren:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap streamlit plotly
   ```
2. **Datensatz wie oben herunterladen und ablegen**
3. **Web-App starten:**
   ```bash
   cd app
   python start_app.py
   ```

### 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Der verwendete Datensatz stammt von Kaggle unter deren Nutzungsbedingungen.

### 🤝 Beitragen

Dieses Repository ist Teil einer Studienarbeit. Feedback und Verbesserungsvorschläge sind willkommen!

---

**Erstellt für:** Duales Studium - Semester 4 - PA2  
**Datum:** August 2025
