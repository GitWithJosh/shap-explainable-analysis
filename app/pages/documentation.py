"""
Dokumentationsseite - Umfassende Dokumentation zur SHAP-basierten Explainable AI Analyse
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from model_utils import get_feature_translations

def render_documentation():
    """Rendert die Dokumentationsseite"""
    st.header("📚 Dokumentation: SHAP Explainable AI")
    
    # Navigation innerhalb der Dokumentation
    doc_section = st.selectbox(
        "Dokumentationsbereich auswählen:",
        ["🎯 Überblick", "🧠 SHAP Grundlagen", "🏗️ Modell & Datensatz", "📊 Visualisierungen"]
    )
    
    if doc_section == "🎯 Überblick":
        render_overview()
    elif doc_section == "🧠 SHAP Grundlagen":
        render_shap_fundamentals()
    elif doc_section == "🏗️ Modell & Datensatz":
        render_model_dataset()
    elif doc_section == "📊 Visualisierungen":
        render_visualizations_guide()

def render_overview():
    """Überblick über die Anwendung"""
    st.subheader("🎯 Überblick")
    st.markdown("""
    Kurz und knapp: Diese App erklärt Modellvorhersagen mit SHAP. Ziel ist, schnell zu verstehen,
    welche Features eine Kreditentscheidung beeinflussen — für Data Scientists und Stakeholder.

    Nutze die Visualisierungen in der App für Detailanalysen; die Notebooks enthalten tiefergehende Erklärungen.
    """)

    # Kurze Architektur-Übersicht
    st.markdown("### 🏗️ System-Architektur (kurz)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Daten:** GiveMeSomeCredit — Bereinigung & Features")
    with col2:
        st.markdown("**Modell:** LightGBM + SHAP (TreeExplainer)")
    with col3:
        st.markdown("**Interface:** Streamlit, interaktive Plots, Export")

def render_shap_fundamentals():
    """SHAP Grundlagen erklären"""
    st.subheader("🧠 SHAP Grundlagen")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) erklärt einzelne Vorhersagen, indem es den Beitrag
    jedes Features zur Vorhersage angibt. Kurz: Baseline + Summe(Shapley-Werte) = Vorhersage.

    Wichtig: SHAP liefert lokale (instanz-spezifische) und globale Einsichten. Verwende mehrere
    Visualisierungen, um ein robustes Bild zu bekommen.
    """)

    st.markdown("### ⚖️ Kurzvergleich zu anderen Methoden")
    comparison_data = {
        'Methode': ['SHAP', 'LIME', 'Permutation Importance'],
        'Scope': ['Lokal + Global', 'Lokal', 'Global'],
        'Modell-agnostisch': ['Teilweise', 'Ja', 'Ja']
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

def render_model_dataset():
    """Modell und Datensatz beschreiben"""
    st.subheader("🏗️ Modell & Datensatz")
    st.markdown("""
    Kurzinformation: Dataset: GiveMeSomeCredit (Kaggle). Ziel ist die Vorhersage von Zahlungsschwierigkeiten.
    Hier ein kompakter Auszug der wichtigsten Features und ihrer Bedeutung.
    """)
    st.markdown("### 🔍 Feature-Übersicht")
    
    feature_info = {
        'Feature (Original)': [
            'RevolvingUtilizationOfUnsecuredLines',
            'age',
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ],
        'Deutsche Übersetzung': [
            'Kreditnutzung ungesicherte Linien',
            'Alter',
            'Anzahl 30-59 Tage verspätete Zahlungen',
            'Schuldenverhältnis',
            'Monatliches Einkommen',
            'Anzahl offener Kreditlinien',
            'Anzahl 90+ Tage verspätete Zahlungen',
            'Anzahl Immobilienkredite',
            'Anzahl 60-89 Tage verspätete Zahlungen',
            'Anzahl Unterhaltsberechtigte'
        ],
        'Datentyp': [
            'Float', 'Integer', 'Integer', 'Float', 'Float',
            'Integer', 'Integer', 'Integer', 'Integer', 'Float'
        ],
        'Wertebereich': [
            '0-1+', '0-109', '0-98', '0-330000', '0-3.008M',
            '0-58', '0-98', '0-54', '0-98', '0-20'
        ]
    }
    
    feature_df = pd.DataFrame(feature_info)
    st.dataframe(feature_df, use_container_width=True)
    
    st.markdown("""
    Feature Engineering (kurz): Imputation, Skalierung, Ausreißerbehandlung und sinnvolle Transformationen.
    Details zu den Schritten stehen in den Notebooks.
    """)
    
    # Modell-Details
    st.markdown("""
    ## 🤖 LightGBM Modell
    
    **Warum LightGBM?**
    - Hohe Performance bei tabellarischen Daten
    - Eingebauter SHAP TreeExplainer Support
    - Effizient bei unbalancierten Datensätzen
    - Gute Regularisierung zur Overfitting-Vermeidung
    
    ### 🎛️ Hyperparameter
    """)
    
    hyperparams = {
        'Parameter': [
            'objective', 'n_estimators', 'max_depth', 'learning_rate',
            'num_leaves', 'min_data_in_leaf', 'scale_pos_weight',
            'subsample', 'colsample_bytree', 'min_split_gain'
        ],
        'Wert': [
            'binary', '400', '5', '0.05',
            '70', '300', '20',
            '0.9', '0.6', '0.15'
        ],
        'Erklärung': [
            'Binäre Klassifikation',
            'Anzahl Boosting-Runden',
            'Maximale Baumtiefe',
            'Lernrate',
            'Maximale Blätter pro Baum',
            'Minimale Samples pro Blatt',
            'Gewichtung für Minority Class',
            'Sampling-Rate für Samples',
            'Sampling-Rate für Features',
            'Minimaler Gain für Split'
        ]
    }
    
    hyperparam_df = pd.DataFrame(hyperparams)
    st.dataframe(hyperparam_df, use_container_width=True)
    
    st.markdown("""
    Performance (Kurz): AUC, Precision und Recall sind verfügbar. Für Entscheidungen immer die
    business-seitigen Kosten (False Positives vs. False Negatives) berücksichtigen.
    """)

def render_visualizations_guide():
    """Visualisierungs-Guide"""
    st.subheader("📊 Visualisierungen verstehen")
    
    # Tab-Navigation für Visualisierungen
    viz_tab = st.selectbox(
        "Visualisierung auswählen:",
        ["🎯 Force Plot", "📊 Bar Chart", "🔧 Waterfall Plot", "📈 Summary Plot", "🎛️ Dependence Plot"]
    )
    
    if viz_tab == "🎯 Force Plot":
        st.markdown("""
        Kurze Erklärung: Der Force Plot zeigt, welche Features die Vorhersage für eine einzelne Instanz erhöhen oder senken.
        Nutze ihn, um schnell zu sehen, welche Faktoren eine Entscheidung getrieben haben.
        """)
    elif viz_tab == "📊 Bar Chart":
        st.markdown("""
        Kurze Erklärung: Bar Charts zeigen die wichtigsten Features (Ranking) für eine Instanz oder das Modell.
        Ideal für einen schnellen Überblick über Feature-Wichtigkeiten.
        """)
    elif viz_tab == "🔧 Waterfall Plot":
        st.markdown("""
        Kurze Erklärung: Waterfall-Plots bauen die Vorhersage schrittweise von der Baseline zur Finalvorhersage auf.
        Gut für Debugging einzelner Entscheidungen.
        """)
    elif viz_tab == "📈 Summary Plot":
        st.markdown("""
        Kurze Erklärung: Summary-Plots zeigen SHAP-Verteilungen über alle Samples und geben globale Einsichten.
        Sie helfen, Muster und Ausreißer zu erkennen.
        """)
    elif viz_tab == "🎛️ Dependence Plot":
        st.markdown("""
        Kurze Erklärung: Dependence-Plots zeigen, wie ein Featurewert den SHAP-Wert beeinflusst (ggf. mit Interaktion).
        Nützlich, um nichtlineare Effekte zu sehen.
        """)
    
    # Allgemeine Interpretationstipps
    st.markdown("""
    ## 💡 Allgemeine Interpretationstipps
    
    ### ✅ Do's
    - Mehrere Visualisierungen kombinieren
    - Kontext der Daten berücksichtigen
    - Business-Logik mit SHAP-Werten abgleichen
    - Unsicherheit kommunizieren
    
    ### ❌ Don'ts
    - Einzelne SHAP-Werte überinterpretieren
    - Kausalität aus Korrelation ableiten
    - SHAP-Werte ohne Kontext bewerten
    - Baseline ignorieren
    
    ### 🔍 Interpretation Checklist
    1. **Baseline verstehen:** Was ist der erwartete Wert?
    2. **Top Features identifizieren:** Welche Features haben den größten Einfluss?
    3. **Plausibilität prüfen:** Machen die Erklärungen Business-Sinn?
    4. **Konsistenz checken:** Sind die Erklärungen über ähnliche Fälle konsistent?
    5. **Unsicherheit bewerten:** Wie konfident ist das Modell?
    """)

if __name__ == "__main__":
    render_documentation()