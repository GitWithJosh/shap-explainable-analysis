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
        ["🎯 Überblick", "🧠 SHAP Grundlagen", "🏗️ Modell & Datensatz", "📊 Visualisierungen", "🔧 Technische Details", "❓ FAQ"]
    )
    
    if doc_section == "🎯 Überblick":
        render_overview()
    elif doc_section == "🧠 SHAP Grundlagen":
        render_shap_fundamentals()
    elif doc_section == "🏗️ Modell & Datensatz":
        render_model_dataset()
    elif doc_section == "📊 Visualisierungen":
        render_visualizations_guide()
    elif doc_section == "🔧 Technische Details":
        render_technical_details()
    elif doc_section == "❓ FAQ":
        render_faq()

def render_overview():
    """Überblick über die Anwendung"""
    st.subheader("🎯 Überblick")
    
    st.markdown("""
    ## Was ist diese Anwendung?
    
    Diese Streamlit-Anwendung demonstriert **Explainable AI (XAI)** durch SHAP-Analysen für Kreditrisikobewertungen.
    Sie ermöglicht es, die Entscheidungen eines Machine Learning-Modells transparent und verständlich zu machen.
    
    ### 🎯 Hauptziele
    
    - **Transparenz:** Verstehe, warum das Modell bestimmte Vorhersagen trifft
    - **Vertrauen:** Baue Vertrauen durch nachvollziehbare Erklärungen auf
    - **Compliance:** Erfülle Anforderungen an erklärbare KI in regulierten Bereichen
    - **Optimierung:** Identifiziere Verbesserungsmöglichkeiten im Modell
    
    ### 🔍 Was kannst du mit der App machen?
    
    1. **Individuelle Analysen:** Untersuche einzelne Kreditentscheidungen im Detail
    2. **Interaktive Visualisierungen:** Nutze verschiedene SHAP-Plots (Force, Bar, Waterfall)
    3. **Batch-Analysen:** Analysiere mehrere Datenpunkte gleichzeitig
    4. **Downloads:** Exportiere Analysen und Visualisierungen
    5. **Filterung:** Fokussiere auf spezielle Fälle (False Positives, etc.)
    
    ### 🏆 Vorteile von Explainable AI
    
    **Für Geschäftsentscheidungen:**
    - Besseres Verständnis der Risikofaktoren
    - Fundierte Kreditentscheidungen
    - Compliance mit regulatorischen Anforderungen
    
    **Für Data Scientists:**
    - Modellvalidierung und -debugging
    - Feature Engineering Insights
    - Bias-Erkennung
    
    **Für Stakeholder:**
    - Vertrauen in KI-Entscheidungen
    - Nachvollziehbare Begründungen
    - Risikomanagement
    """)
    
    # Architektur-Diagramm
    st.markdown("### 🏗️ System-Architektur")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Datenebene**
        - Give Me Some Credit Datensatz
        - Feature Engineering
        - Datenvalidierung
        """)
    
    with col2:
        st.markdown("""
        **🤖 Modellebene**
        - LightGBM Classifier
        - SHAP TreeExplainer
        - Vorhersage-Pipeline
        """)
    
    with col3:
        st.markdown("""
        **🖥️ Interface-Ebene**
        - Streamlit Web-App
        - Interaktive Visualisierungen
        - Download-Funktionen
        """)

def render_shap_fundamentals():
    """SHAP Grundlagen erklären"""
    st.subheader("🧠 SHAP Grundlagen")
    
    st.markdown("""
    ## Was ist SHAP?
    
    **SHAP (SHapley Additive exPlanations)** ist eine moderne Methode zur Erklärung von Machine Learning-Modellen,
    die auf der Spieltheorie basiert und faire, konsistente Erklärungen liefert.
    
    ### 🎯 Kernkonzepte
    
    **Shapley Values:**
    - Stammen aus der kooperativen Spieltheorie
    - Messen den marginalen Beitrag jedes Features
    - Erfüllen mathematische Fairness-Axiome
    
    **Additive Eigenschaft:**
    ```
    f(x) = E[f(X)] + Σ φᵢ
    ```
    - f(x) = Modellvorhersage
    - E[f(X)] = Erwarteter Wert (Baseline)
    - φᵢ = SHAP-Wert für Feature i
    """)
    
    # Interaktives SHAP Beispiel
    st.markdown("### 📊 Interaktives SHAP Beispiel")
    
    # Beispiel-Daten erstellen
    baseline = 0.3
    features = ['Alter', 'Einkommen', 'Schulden', 'Kredithistorie']
    shap_values = [0.15, -0.08, 0.12, -0.05]
    
    # Waterfall-ähnliche Visualisierung mit Plotly
    cumulative = [baseline]
    for val in shap_values:
        cumulative.append(cumulative[-1] + val)
    
    fig = go.Figure()
    
    # Baseline
    fig.add_trace(go.Bar(
        x=['Baseline'],
        y=[baseline],
        name='Baseline',
        marker_color='gray'
    ))
    
    # SHAP Beiträge
    for i, (feature, shap_val) in enumerate(zip(features, shap_values)):
        color = 'red' if shap_val > 0 else 'green'
        fig.add_trace(go.Bar(
            x=[feature],
            y=[shap_val],
            name=f'{feature}: {shap_val:+.3f}',
            marker_color=color,
            base=cumulative[i]
        ))
    
    # Finale Vorhersage
    fig.add_trace(go.Bar(
        x=['Vorhersage'],
        y=[cumulative[-1]],
        name='Finale Vorhersage',
        marker_color='blue'
    ))
    
    fig.update_layout(
        title='SHAP Additive Erklärung (Beispiel)',
        yaxis_title='Vorhersagewert',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    ### 🔍 Interpretation
    
    **Positive SHAP-Werte (rot):**
    - Erhöhen die Vorhersage
    - Deuten auf höheres Kreditrisiko hin
    - Beispiel: Hohes Alter oder hohe Schulden
    
    **Negative SHAP-Werte (grün):**
    - Verringern die Vorhersage
    - Deuten auf niedrigeres Kreditrisiko hin
    - Beispiel: Hohes Einkommen oder gute Kredithistorie
    
    ### ✅ SHAP Axiome
    
    1. **Effizienz:** Summe aller SHAP-Werte = Vorhersage - Baseline
    2. **Symmetrie:** Features mit gleichem marginalen Beitrag haben gleiche SHAP-Werte
    3. **Dummy:** Features ohne Einfluss haben SHAP-Wert = 0
    4. **Additivität:** Für zusammengesetzte Modelle addieren sich die SHAP-Werte
    """)
    
    # Vergleich mit anderen XAI-Methoden
    st.markdown("### ⚖️ Vergleich mit anderen XAI-Methoden")
    
    comparison_data = {
        'Methode': ['SHAP', 'LIME', 'Permutation Importance', 'Feature Importance'],
        'Lokal/Global': ['Beide', 'Lokal', 'Global', 'Global'],
        'Modell-agnostisch': ['Teilweise', 'Ja', 'Ja', 'Nein'],
        'Axiomatisch fundiert': ['Ja', 'Nein', 'Nein', 'Nein'],
        'Konsistenz': ['Hoch', 'Medium', 'Medium', 'Hoch'],
        'Rechenaufwand': ['Hoch', 'Medium', 'Hoch', 'Niedrig']
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

def render_model_dataset():
    """Modell und Datensatz beschreiben"""
    st.subheader("🏗️ Modell & Datensatz")
    
    # Datensatz-Information
    st.markdown("""
    ## 📊 Give Me Some Credit Datensatz
    
    **Quelle:** [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/)
    
    **Ziel:** Vorhersage der Wahrscheinlichkeit, dass eine Person in den nächsten zwei Jahren
    finanzielle Schwierigkeiten haben wird.
    
    ### 📈 Datensatz-Statistiken
    
    - **Trainings-Samples:** ~150,000
    - **Test-Samples:** ~101,503  
    - **Features:** 11 (10 numerische + 1 kategorische)
    - **Zielklassen:** Binär (0 = Good Risk, 1 = Bad Risk)
    - **Class Imbalance:** ~93% Good Risk, ~7% Bad Risk
    """)
    
    # Feature-Übersicht
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
    
    # Feature Engineering
    st.markdown("""
    ### ⚙️ Feature Engineering
    
    **Angewandte Transformationen:**
    
    1. **Box-Cox Transformation:** Für schief verteilte numerische Features
    2. **Missing Value Imputation:** Median für numerische, Modus für kategorische
    3. **Outlier Treatment:** Caps bei 99.5% Percentile
    4. **Feature Scaling:** StandardScaler für LightGBM
    
    **Zusätzliche Features:**
    - Ratios und Interaktionen zwischen bestehenden Features
    - Binning von kontinuierlichen Variablen
    - Polynomial Features für nichtlineare Beziehungen
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
    
    # Performance-Metriken
    st.markdown("""
    ### 📊 Modell-Performance
    
    **Evaluationsmetriken:**
    - **Accuracy:** ~93% (aber misleading bei unbalancierten Daten)
    - **Precision:** ~85% (von vorhergesagten Bad Risks sind 85% tatsächlich bad)
    - **Recall:** ~42% (von tatsächlichen Bad Risks werden 42% erkannt)
    - **F1-Score:** ~56% (harmonisches Mittel aus Precision und Recall)
    - **AUC-ROC:** ~0.85 (sehr gute Trennfähigkeit)
    
    **Business-Metriken:**
    - **Cost-Benefit-Analyse:** Berücksichtigung der Kosten von False Negatives vs. False Positives
    - **Expected Loss:** Erwarteter Verlust bei verschiedenen Schwellenwerten
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
        ## 🎯 Force Plot
        
        **Was zeigt er?**
        - Horizontale Darstellung der SHAP-Beiträge
        - Rote Bereiche = Erhöhen die Vorhersage
        - Blaue Bereiche = Verringern die Vorhersage
        
        **Wann verwenden?**
        - Erklärung einzelner Vorhersagen
        - Schnelle visuelle Übersicht
        - Präsentationen für Stakeholder
        
        **Interpretation:**
        - Baseline (graue Linie) = erwarteter Modelloutput
        - Endpunkt = tatsächliche Vorhersage für diese Instanz
        - Breite der Bereiche = Einfluss des Features
        """)
        
        # Beispiel-Force-Plot (vereinfacht)
        fig = go.Figure()
        
        # Baseline
        fig.add_shape(
            type="line",
            x0=0, y0=0.3, x1=10, y1=0.3,
            line=dict(color="gray", width=3, dash="dash")
        )
        
        # Features
        features_example = [
            ("Alter: 35", 2, 0.05, "red"),
            ("Einkommen: 5000€", 3, -0.08, "blue"),
            ("Schulden: 30%", 2, 0.12, "red"),
            ("Historie: gut", 3, -0.04, "blue")
        ]
        
        y_pos = 0.3
        for name, width, contribution, color in features_example:
            fig.add_shape(
                type="rect",
                x0=len(name), y0=y_pos-0.02, 
                x1=len(name)+width, y1=y_pos+0.02,
                fillcolor=color, opacity=0.7
            )
            y_pos += contribution
        
        fig.update_layout(
            title="Force Plot Aufbau (schematisch)",
            xaxis_title="Features",
            yaxis_title="Vorhersagewert",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_tab == "📊 Bar Chart":
        st.markdown("""
        ## 📊 Bar Chart
        
        **Was zeigt er?**
        - Ranking der wichtigsten Features
        - SHAP-Werte als horizontale Balken
        - Positive Werte = rot, negative Werte = grün
        
        **Wann verwenden?**
        - Top-Features identifizieren
        - Einfache Feature-Wichtigkeit
        - Vergleich zwischen Instanzen
        
        **Interpretation:**
        - Längere Balken = größerer Einfluss
        - Rote Balken = erhöhen Risiko
        - Grüne Balken = verringern Risiko
        """)
        
        # Beispiel Bar Chart
        example_features = ['Schulden_Ratio', 'Verspätungen_90d', 'Alter', 'Einkommen', 'Kreditlinien']
        example_shap_values = [0.15, 0.12, -0.08, -0.06, 0.03]
        
        colors = ['red' if val > 0 else 'green' for val in example_shap_values]
        
        fig = go.Figure(go.Bar(
            x=example_shap_values,
            y=example_features,
            orientation='h',
            marker_color=colors,
            text=[f'{val:.3f}' for val in example_shap_values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title='SHAP Bar Chart (Beispiel)',
            xaxis_title='SHAP-Wert',
            yaxis_title='Features',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_tab == "🔧 Waterfall Plot":
        st.markdown("""
        ## 🔧 Waterfall Plot
        
        **Was zeigt er?**
        - Schrittweise Aufbauung der Vorhersage
        - Jeder Balken = Beitrag eines Features
        - Start bei Baseline, Ende bei finaler Vorhersage
        
        **Wann verwenden?**
        - Detaillierte Schritt-für-Schritt Erklärung
        - Verstehen der Entscheidungslogik
        - Debugging von Modellverhalten
        
        **Interpretation:**
        - Start = Expected Value (Baseline)
        - Jeder Schritt = SHAP-Beitrag eines Features
        - Ende = Finale Modellvorhersage
        """)
    
    elif viz_tab == "📈 Summary Plot":
        st.markdown("""
        ## 📈 Summary Plot
        
        **Was zeigt er?**
        - SHAP-Werte für alle Features über alle Samples
        - Jeder Punkt = eine Vorhersage
        - Farbe = Feature-Wert (niedrig bis hoch)
        
        **Wann verwenden?**
        - Globale Feature-Wichtigkeit verstehen
        - Patterns und Trends identifizieren
        - Feature-Interaktionen erkennen
        
        **Interpretation:**
        - Y-Achse = Features (sortiert nach Wichtigkeit)
        - X-Achse = SHAP-Wert
        - Farbe = Feature-Wert
        - Dichte = Verteilung der SHAP-Werte
        """)
    
    elif viz_tab == "🎛️ Dependence Plot":
        st.markdown("""
        ## 🎛️ Dependence Plot
        
        **Was zeigt er?**
        - Beziehung zwischen Feature-Wert und SHAP-Wert
        - Scatterplot mit Feature-Wert auf X-Achse
        - Farbe zeigt Interaktions-Feature
        
        **Wann verwenden?**
        - Nichtlineare Beziehungen verstehen
        - Feature-Interaktionen analysieren
        - Schwellenwerte identifizieren
        
        **Interpretation:**
        - X-Achse = Feature-Wert
        - Y-Achse = SHAP-Wert für dieses Feature
        - Farbe = Wert des Interaktions-Features
        - Trend = Richtung des Feature-Einflusses
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

def render_technical_details():
    """Technische Details"""
    st.subheader("🔧 Technische Details")
    
    tech_section = st.selectbox(
        "Technischer Bereich:",
        ["⚙️ Implementierung", "🚀 Performance", "🔒 Sicherheit", "📦 Dependencies", "🐛 Debugging"]
    )
    
    if tech_section == "⚙️ Implementierung":
        st.markdown("""
        ## ⚙️ Implementierung
        
        ### 🏗️ Architektur
        
        **Modulstruktur:**
        ```
        app/
        ├── streamlit_app.py          # Haupt-App (Navigation)
        ├── pages/
        │   ├── shap_analysis.py      # SHAP Analyse Seite
        │   └── documentation.py      # Diese Dokumentation
        ├── model_utils.py            # Modell-Utilities
        ├── model_components.pkl      # Gespeicherte Modellkomponenten
        └── requirements.txt          # Dependencies
        ```
        
        **Caching-Strategie:**
        - `@st.cache_data` für Modellkomponenten
        - `@st.cache_resource` für große Objekte
        - Automatische Cache-Invalidierung bei Änderungen
        
        **Session State Management:**
        - URL-Parameter für Navigation
        - Persistent state für Benutzerinteraktionen
        - Optimierte Re-runs
        """)
        
        # Code-Beispiel
        st.code("""
        @st.cache_data
        def load_model_components():
            \"\"\"Lädt Modellkomponenten mit Caching\"\"\"
            with open('model_components.pkl', 'rb') as f:
                return pickle.load(f)
        
        # SHAP TreeExplainer für LightGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        """, language='python')
    
    elif tech_section == "🚀 Performance":
        st.markdown("""
        ## 🚀 Performance-Optimierung
        
        ### ⚡ Ladezeiten
        - **Model Loading:** ~2-3 Sekunden (cached)
        - **SHAP Calculation:** Pre-computed für Test-Set
        - **Visualisierung:** ~0.5 Sekunden pro Plot
        - **Page Navigation:** ~0.1 Sekunden
        
        ### 💾 Memory Management
        - Modellkomponenten: ~50MB RAM
        - SHAP Values: ~20MB RAM  
        - Visualisierungen: ~5MB RAM
        - Total: ~75MB RAM
        
        ### 🔄 Optimierungsstrategien
        1. **Pre-computation:** SHAP-Werte vorberechnet
        2. **Lazy Loading:** Visualisierungen on-demand
        3. **Compression:** Pickle mit höchstem Compression-Level
        4. **Streaming:** Große DataFrames chunked laden
        """)
        
        # Performance-Metriken
        perf_data = {
            'Operation': ['Model Load', 'SHAP Calculation', 'Force Plot', 'Bar Chart', 'Waterfall Plot'],
            'Zeit (ms)': [2500, 50, 300, 200, 400],
            'Memory (MB)': [45, 5, 2, 1, 3],
            'Cached': ['Ja', 'Ja', 'Nein', 'Nein', 'Nein']
        }
        
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)
    
    elif tech_section == "🔒 Sicherheit":
        st.markdown("""
        ## 🔒 Sicherheit & Datenschutz
        
        ### 🛡️ Datensicherheit
        - **Kein External Data Transfer:** Alle Berechnungen lokal
        - **No User Data Storage:** Keine Persistierung von Benutzerdaten
        - **Secure Defaults:** Streamlit Security Best Practices
        
        ### 🔐 Access Control
        - **Local Deployment:** App läuft nur lokal
        - **No Authentication Required:** Für Demo-Zwecke
        - **Port-based Access:** Zugriff nur über localhost:8501
        
        ### 📋 Compliance
        - **GDPR Ready:** Keine personenbezogenen Daten gespeichert
        - **Audit Trail:** Streamlit Logging verfügbar
        - **Data Lineage:** Nachvollziehbare Modell-Pipeline
        
        ### ⚠️ Sicherheitshinweise
        1. **Produktions-Deployment:** Authentication implementieren
        2. **Sensitive Data:** Nie echte Kundendaten verwenden
        3. **Network Security:** Firewall-Konfiguration beachten
        4. **Updates:** Dependencies regelmäßig aktualisieren
        """)
    
    elif tech_section == "📦 Dependencies":
        st.markdown("""
        ## 📦 Dependencies & Installation
        
        ### 🐍 Python Version
        - **Minimum:** Python 3.8+
        - **Recommended:** Python 3.9 oder 3.10
        - **Tested:** Python 3.11
        
        ### 📚 Core Dependencies
        """)
        
        deps_data = {
            'Package': ['streamlit', 'shap', 'lightgbm', 'pandas', 'numpy', 'matplotlib', 'plotly', 'scikit-learn'],
            'Version': ['>=1.28.0', '>=0.42.0', '>=4.0.0', '>=1.5.0', '>=1.21.0', '>=3.5.0', '>=5.15.0', '>=1.3.0'],
            'Zweck': [
                'Web-App Framework',
                'SHAP Explanations',
                'Machine Learning Model',
                'Data Manipulation',
                'Numerical Computing',
                'Static Plots',
                'Interactive Plots',
                'ML Utilities'
            ],
            'Größe': ['~15MB', '~8MB', '~5MB', '~12MB', '~20MB', '~30MB', '~25MB', '~28MB']
        }
        
        deps_df = pd.DataFrame(deps_data)
        st.dataframe(deps_df, use_container_width=True)
        
        st.markdown("""
        ### 📥 Installation
        
        **Via pip:**
        ```bash
        pip install -r requirements.txt
        ```
        
        **Via conda:**
        ```bash
        conda install streamlit shap lightgbm pandas numpy matplotlib plotly scikit-learn
        ```
        
        **Development Setup:**
        ```bash
        # Virtual Environment erstellen
        python -m venv shap_env
        source shap_env/bin/activate  # Linux/Mac
        # shap_env\\Scripts\\activate  # Windows
        
        # Dependencies installieren
        pip install -r requirements.txt
        
        # App starten
        streamlit run streamlit_app.py
        ```
        """)
    
    elif tech_section == "🐛 Debugging":
        st.markdown("""
        ## 🐛 Debugging & Troubleshooting
        
        ### ❌ Häufige Fehler
        
        **1. "Model components not found"**
        ```
        Lösung: Stelle sicher, dass model_components.pkl existiert
        - Führe zuerst das Training-Notebook aus
        - Prüfe den app/ Ordner auf die Datei
        ```
        
        **2. "SHAP explainer error"**
        ```
        Lösung: Version-Kompatibilität prüfen
        - shap >= 0.42.0
        - lightgbm >= 4.0.0
        - Neu installieren: pip install --upgrade shap lightgbm
        ```
        
        **3. "Streamlit port already in use"**
        ```
        Lösung: Port ändern oder freigeben
        - streamlit run app.py --server.port 8502
        - Oder anderen Streamlit-Prozess beenden
        ```
        
        ### 🔍 Debug-Modi
        
        **Streamlit Debug Mode:**
        ```bash
        streamlit run app.py --logger.level debug
        ```
        
        **Python Debug Mode:**
        ```python
        import logging
        logging.basicConfig(level=logging.DEBUG)
        ```
        
        ### 📊 Monitoring
        - **Resource Usage:** Task Manager / htop
        - **Network Traffic:** Streamlit Analytics
        - **Error Logs:** Terminal Output
        - **Performance:** Streamlit Profiler
        
        ### 🆘 Support
        1. **Check Logs:** Terminal Output analysieren
        2. **Version Check:** pip list | grep -E "(streamlit|shap|lightgbm)"
        3. **Environment:** Virtual Environment aktiviert?
        4. **Permissions:** Schreibrechte für Cache-Ordner?
        """)

def render_faq():
    """FAQ Sektion"""
    st.subheader("❓ Häufig gestellte Fragen")
    
    faqs = [
        {
            "question": "🤔 Was bedeuten die SHAP-Werte genau?",
            "answer": """
            SHAP-Werte messen den marginalen Beitrag jedes Features zur Vorhersage. 
            Ein positiver SHAP-Wert von +0.15 für 'Alter' bedeutet, dass das Alter dieser Person
            die Kreditrisiko-Wahrscheinlichkeit um 0.15 (im Logit-Raum) erhöht.
            
            **Wichtig:** SHAP-Werte sind additiv:
            Baseline + Summe aller SHAP-Werte = Finale Vorhersage
            """
        },
        {
            "question": "📊 Warum unterscheiden sich die SHAP-Werte zwischen ähnlichen Personen?",
            "answer": """
            SHAP-Werte sind instanz-spezifisch und berücksichtigen:
            - **Interaktionen:** Features beeinflussen sich gegenseitig
            - **Nichtlinearität:** Der gleiche Feature-Wert kann je nach Kontext unterschiedlich wirken
            - **Baseline:** Vergleich zum durchschnittlichen Datenpunkt
            
            Kleine Unterschiede in anderen Features können große SHAP-Unterschiede verursachen.
            """
        },
        {
            "question": "⚖️ Kann ich SHAP-Werte für Fairness-Analysen verwenden?",
            "answer": """
            SHAP-Werte können bei Fairness-Analysen helfen, sind aber nicht ausreichend:
            
            **Hilfreich für:**
            - Identifikation diskriminierender Features
            - Verstehen von Bias-Quellen
            - Erklärung unterschiedlicher Behandlung
            
            **Limitationen:**
            - Zeigen nur Korrelationen, nicht Kausalität
            - Proxy-Diskriminierung schwer erkennbar
            - Benötigen zusätzliche statistische Tests
            """
        },
        {
            "question": "🎯 Wie genau sind die SHAP-Erklärungen?",
            "answer": """
            SHAP-Erklärungen sind mathematisch exakt für das gegebene Modell:
            - **TreeExplainer:** Exakte Berechnung für Baum-basierte Modelle
            - **Approximations-Fehler:** Vernachlässigbar bei LightGBM
            - **Konsistenz:** Erfüllt alle Shapley-Axiome
            
            **Aber:** Erklärungen sind nur so gut wie das Modell selbst.
            Schlechte Modelle führen zu irreführenden Erklärungen.
            """
        },
        {
            "question": "💼 Wie kann ich SHAP in der Praxis einsetzen?",
            "answer": """
            **Kreditrisiko:**
            - Erklärung von Kreditablehnungen
            - Validierung von Risikofaktoren
            - Regulatorische Compliance
            
            **Model Governance:**
            - Modellvalidierung und -monitoring
            - Bias-Erkennung und -mitigation
            - Feature Engineering Insights
            
            **Business Intelligence:**
            - Identifikation von Risikotreibern
            - Strategische Entscheidungsunterstützung
            - Kundensegmentierung
            """
        },
        {
            "question": "🔧 Welche SHAP-Visualisierung soll ich wann verwenden?",
            "answer": """
            **Force Plot:** 
            - Einzelne Vorhersage erklären
            - Stakeholder-Präsentationen
            - Schnelle Übersicht
            
            **Waterfall Plot:**
            - Detaillierte Schritt-für-Schritt Erklärung
            - Model Debugging
            - Audits
            
            **Bar Chart:**
            - Feature Ranking
            - Vergleiche zwischen Instanzen
            - Top-N Faktoren
            
            **Summary Plot:**
            - Globale Patterns verstehen
            - Feature-Verteilungen analysieren
            - Model-weite Trends
            """
        },
        {
            "question": "⚠️ Was sind die Limitationen von SHAP?",
            "answer": """
            **Technische Limitationen:**
            - Rechenaufwendig bei großen Datensätzen
            - Approximationsfehler bei komplexen Modellen
            - Korrelierte Features können problematisch sein
            
            **Interpretations-Limitationen:**
            - Zeigt 'Was' aber nicht 'Warum'
            - Kausalität vs. Korrelation
            - Kontextwissen erforderlich
            
            **Praktische Limitationen:**
            - Benötigt ML-Verständnis
            - Kann überkomplex für Endnutzer sein
            - Nicht für alle Modelltypen optimal
            """
        }
    ]
    
    for i, faq in enumerate(faqs):
        with st.expander(faq["question"]):
            st.markdown(faq["answer"])
    
    # Zusätzliche Ressourcen
    st.markdown("""
    ## 📚 Zusätzliche Ressourcen
    
    **SHAP Dokumentation:**
    - [Offizielle SHAP Docs](https://shap.readthedocs.io/)
    - [SHAP GitHub](https://github.com/slundberg/shap)
    - [Shapley Values Paper](https://christophm.github.io/interpretable-ml-book/shapley.html)
    
    **Explainable AI:**
    - [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
    - [Google's AI Explanations](https://cloud.google.com/ai-platform/prediction/docs/ai-explanations)
    - [Microsoft InterpretML](https://interpret.ml/)
    
    **Business Applications:**
    - [Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)
    - [GDPR Right to Explanation](https://gdpr.eu/right-to-explanation/)
    - [Algorithmic Accountability Act](https://www.congress.gov/bill/116th-congress/house-bill/2231)
    """)

if __name__ == "__main__":
    render_documentation()