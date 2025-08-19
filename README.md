# SHAP-basierte Explainable AI Analyse

## Implementierung für Kreditrisikobewertung

Dieses Repository enthält eine umfassende Analyse zur Erklärung von Machine Learning-Vorhersagen im Kontext der Kreditrisikobewertung mit SHAP (SHapley Additive exPlanations).

### 📊 Datensatz

> ⚠️ **Hinweis:** Der Datensatz ist aus Lizenzgründen nicht im Repository enthalten. Sie müssen ihn separat von Kaggle herunterladen (siehe Quick Start Anweisungen).

**[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/overview)** - Kaggle Competition Datensatz

### 📋 Variable Beschreibung

| Variable Name | Beschreibung |
|---------------|--------------|
| SeriousDlqin2yrs (Target) | Person hatte 90+ Tage Zahlungsrückstand oder schlimmer |
| RevolvingUtilizationOfUnsecuredLines | Gesamtsaldo auf Kreditkarten und persönlichen Kreditlinien geteilt durch Kreditlimits |
| age | Alter des Kreditnehmers in Jahren |
| NumberOfTime30-59DaysPastDueNotWorse | Anzahl der 30-59 Tage Zahlungsrückstände in den letzten 2 Jahren |
| DebtRatio | Monatliche Schuldenzahlungen, Unterhalt, Lebenshaltungskosten geteilt durch monatliches Bruttoeinkommen |
| MonthlyIncome | Monatliches Einkommen |
| NumberOfOpenCreditLinesAndLoans | Anzahl offener Kredite und Kreditlinien |
| NumberOfTimes90DaysLate | Anzahl der 90+ Tage Zahlungsrückstände |
| NumberRealEstateLoansOrLines | Anzahl Hypotheken- und Immobilienkredite |
| NumberOfTime60-89DaysPastDueNotWorse | Anzahl der 60-89 Tage Zahlungsrückstände in den letzten 2 Jahren |
| NumberOfDependents | Anzahl Abhängige in der Familie |

### 🎯 Methodik

Diese Analyse verwendet **SHAP (SHapley Additive exPlanations)** zur Erklärung von Machine Learning-Vorhersagen. Das implementierte **LightGBM-Modell** wird durch verschiedene XAI-Techniken interpretierbar gemacht:

- **Feature Importance Analyse** mit SHAP Values
- **Waterfall Plots** für individuelle Vorhersageerklärungen
- **False Positive/Negative Analyse** zur Modellvalidierung
- **Wahrscheinlichkeits- und Logit-Raum Visualisierungen**


⚠️ **Wichtig:** Der Datensatz ist aus Lizenzgründen nicht im Repository enthalten und muss separat heruntergeladen werden.

1. **Repository klonen:**
   ```bash
   git clone https://github.com/GitWithJosh/shap-explainable-analysis.git
   cd shap_demonstration
   ```

2. **Datensatz herunterladen:**
   - Besuchen Sie die [Give Me Some Credit Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data)
   - Melden Sie sich bei Kaggle an (kostenlos)
   - Laden Sie die folgenden Dateien herunter:
     - `cs-training.csv`
     - `cs-test.csv`
     - `Data Dictionary.xls`
     - `sampleEntry.csv`
   - Erstellen Sie einen Ordner namens `GiveMeSomeCredit/` im Projektverzeichnis
   - Legen Sie alle heruntergeladenen Dateien in diesen Ordner

3. **Abhängigkeiten installieren:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn lightgbm shap jupyter
   ```

4. **Jupyter Notebook starten:**
   ```bash
   jupyter notebook shap_demonstration.ipynb

### 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Der verwendete Datensatz stammt von Kaggle unter deren Nutzungsbedingungen.

### 🤝 Beitragen

Dieses Repository ist Teil einer Studienarbeit. Feedback und Verbesserungsvorschläge sind willkommen!

---

**Erstellt für:** Duales Studium - Semester 4 - PA2  
**Datum:** August 2025
