"""
SHAP Analyse Seite - Interaktive SHAP-Visualisierungen für Kreditrisikobewertung
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from model_utils import load_model_components, get_feature_translations
import io
import base64
import warnings
warnings.filterwarnings('ignore')

def create_download_link(data, filename, file_type="csv"):
    """Erstellt einen Download-Link für Daten"""
    if file_type == "csv":
        csv_data = data.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 {filename} herunterladen</a>'
    elif file_type == "png":
        img_buffer = io.BytesIO()
        data.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        b64 = base64.b64encode(img_buffer.getvalue()).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 {filename} herunterladen</a>'
    return href

def highlight_shap(val):
    """Farbkodierung für SHAP-Werte"""
    if isinstance(val, (int, float)):
        color = 'red' if val > 0 else 'green' if val < 0 else 'black'
        return f'color: {color}'
    return ''

@st.cache_data
def load_cached_data():
    """Lädt die Modellkomponenten mit Caching"""
    return load_model_components()

def render_shap_analysis():
    """Rendert die SHAP Analyse Seite"""
    st.header("🔍 Interaktive SHAP Analyse")
    st.markdown("""
    Analysiere individuelle Vorhersagen des Kreditrisiko-Modells mit SHAP-Erklärungen.
    Wähle verschiedene Datenpunkte aus und verstehe, welche Faktoren die Entscheidung beeinflussen.
    """)

    # Lade Modellkomponenten
    with st.spinner('Lade Modellkomponenten...'):
        try:
            model_components = load_cached_data()
        except Exception as e:
            st.error(f"Fehler beim Laden der Modellkomponenten: {str(e)}")
            st.info("Stelle sicher, dass das Modell trainiert wurde und die model_components.pkl Datei existiert.")
            return

    # Extrahiere Komponenten
    explainer = model_components['explainer']
    x_test = model_components['x_test']
    x_test_original = model_components['x_test_original']
    y_test = model_components['y_test']
    y_pred = model_components['y_pred']
    y_pred_proba = model_components['y_pred_proba']
    shap_values_test = model_components['shap_values_test']
    feature_names = model_components['feature_names']
    expected_value = model_components['expected_value']

    # Feature-Übersetzungen
    feature_translations = get_feature_translations()

    # Sidebar Konfiguration
    st.sidebar.header("⚙️ Analyse-Konfiguration")
    
    # Filter für spezielle Fälle
    case_filter = st.sidebar.selectbox(
        "Filter für spezielle Fälle:",
        options=[
            "Alle Datenpunkte",
            "True Positives (korrekt als Risiko erkannt)",
            "True Negatives (korrekt als kein Risiko erkannt)",
            "False Positives (fälschlicherweise als Risiko erkannt)",
            "False Negatives (übersehenes Risiko)"
        ]
    )

    # Filtere Indizes basierend auf Auswahl
    if case_filter == "Alle Datenpunkte":
        filtered_indices = np.arange(len(y_test))
    elif case_filter == "True Positives (korrekt als Risiko erkannt)":
        filtered_indices = np.where((y_pred == 1) & (y_test == 1))[0]
    elif case_filter == "True Negatives (korrekt als kein Risiko erkannt)":
        filtered_indices = np.where((y_pred == 0) & (y_test == 0))[0]
    elif case_filter == "False Positives (fälschlicherweise als Risiko erkannt)":
        filtered_indices = np.where((y_pred == 1) & (y_test == 0))[0]
    elif case_filter == "False Negatives (übersehenes Risiko)":
        filtered_indices = np.where((y_pred == 0) & (y_test == 1))[0]
    else:
        filtered_indices = np.arange(len(y_test))

    # Index-Auswahl basierend auf Filter
    num_options = len(filtered_indices)
    if num_options > 0:
        st.sidebar.info(f"**{num_options}** Datenpunkte gefunden für: {case_filter}")
        
        # Bei wenigen Optionen: Dropdown, sonst Slider
        if num_options <= 20:
            selected_index = st.sidebar.selectbox(
                f"Wähle Index:",
                options=filtered_indices,
                format_func=lambda x: f"Index {x}"
            )
        else:
            # Position management
            query_params = st.query_params
            current_pos = int(query_params.get("pos", 0))
            
            if current_pos >= num_options or current_pos < 0:
                current_pos = 0

            # Quick Actions
            st.sidebar.markdown("**🎯 Quick Actions:**")
            col_quick1, col_quick2, col_quick3 = st.sidebar.columns(3)
            
            with col_quick1:
                if st.button("⏮️", help="Erste Position"):
                    st.query_params["pos"] = 0
                    st.rerun()
            
            with col_quick2:
                if st.button("🎲", help="Zufällige Position"):
                    random_pos = np.random.randint(0, num_options)
                    st.query_params["pos"] = str(random_pos)
                    st.rerun()
            
            with col_quick3:
                if st.button("⏭️", help="Letzte Position"):
                    st.query_params["pos"] = str(num_options - 1)
                    st.rerun()

            # Position Slider
            position = st.sidebar.slider(
                "Position auswählen:",
                min_value=0,
                max_value=num_options - 1,
                value=current_pos,
                help=f"Navigiere durch die {num_options} gefilterten Datenpunkte"
            )
            
            if position != current_pos:
                st.query_params["pos"] = str(position)
                st.rerun()

            selected_index = filtered_indices[current_pos]
            st.sidebar.success(f"**Position {current_pos + 1}/{num_options}** - Index: {selected_index}")
    else:
        st.sidebar.warning(f"Keine Datenpunkte für {case_filter} gefunden.")
        selected_index = 0

    # Hauptanalyse
    st.divider()
    
    # Basis-Informationen
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Modell-Vorhersage")
        
        # Extrahiere Daten für ausgewählten Index
        actual_class = y_test.iloc[selected_index]
        predicted_probability = y_pred_proba[selected_index]
        predicted_class = y_pred[selected_index]
        
        # Status-Bestimmung
        if actual_class == 1 and predicted_class == 1:
            status = "True Positive ✅"
            status_color = "green"
        elif actual_class == 0 and predicted_class == 0:
            status = "True Negative ✅"
            status_color = "green"
        elif actual_class == 0 and predicted_class == 1:
            status = "False Positive ❌"
            status_color = "red"
        else:
            status = "False Negative ❌"
            status_color = "orange"
        
        # Metriken anzeigen
        st.metric("Index", selected_index)
        st.metric("Tatsächliche Klasse", "Bad Risk (1)" if actual_class == 1 else "Good Risk (0)")
        st.metric("Modell-Vorhersage", f"{predicted_probability:.4f}")
        st.metric("Modell-Entscheidung", "Bad Risk" if predicted_class == 1 else "Good Risk")
        
        # Status mit Farbe
        st.markdown(f"**Status:** <span style='color: {status_color}'>{status}</span>", unsafe_allow_html=True)

    with col2:
        st.subheader("📈 Wahrscheinlichkeits-Visualisierung")
        
        # Gauge Chart für Wahrscheinlichkeit
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = predicted_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risiko-Wahrscheinlichkeit (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # SHAP Visualisierungen
    st.subheader("💧 SHAP Visualisierungen")

    # Tabs für verschiedene Visualisierungen
    tab1, tab2, tab3 = st.tabs(["🎯 Force Plot", "📊 Bar Chart", "🔧 Waterfall Plot"])

    with tab1:
        # SHAP Force Plot
        try:
            st.write("**Force Plot Interpretation:**")
            st.write("- Rote Bereiche erhöhen die Vorhersage (Richtung höheres Risiko)")
            st.write("- Blaue Bereiche verringern die Vorhersage (Richtung geringeres Risiko)")
            
            # Force Plot erstellen (als HTML)
            force_plot = shap.force_plot(
                expected_value, 
                shap_values_test[selected_index], 
                x_test_original.iloc[selected_index],
                feature_names=feature_names,
                matplotlib=False
            )
            
            # HTML ausgeben
            shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
            components.html(shap_html, height=300)
            
            # Download-Option für Force Plot
            with st.expander("📥 Force Plot herunterladen"):
                try:
                    fig_force = plt.figure(figsize=(16, 4))
                    shap.force_plot(
                        expected_value, 
                        shap_values_test[selected_index], 
                        x_test_original.iloc[selected_index],
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    download_link = create_download_link(fig_force, f"force_plot_index_{selected_index}.png", "png")
                    st.markdown(download_link, unsafe_allow_html=True)
                    plt.close(fig_force)
                except:
                    st.warning("Download für Force Plot nicht verfügbar.")
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen des SHAP Force Plots: {str(e)}")

    with tab2:
        # SHAP Bar Chart mit Plotly
        try:
            # Bereite Daten für Bar Chart vor
            shap_abs = np.abs(shap_values_test[selected_index])
            top_indices = np.argsort(shap_abs)[-10:]  # Top 10 Features
            
            top_features = [feature_names[i] for i in top_indices]
            top_features_de = [feature_translations.get(f, f) for f in top_features]
            top_shap_values = [shap_values_test[selected_index][i] for i in top_indices]
            
            # Erstelle Plotly Bar Chart
            colors = ['red' if val > 0 else 'green' for val in top_shap_values]
            
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=top_shap_values,
                    y=top_features_de,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{val:.4f}' for val in top_shap_values],
                    textposition='auto',
                )
            ])
            
            fig_bar.update_layout(
                title=f'Top 10 SHAP-Beiträge - Index: {selected_index}',
                xaxis_title='SHAP-Beitrag',
                yaxis_title='Features',
                height=500,
                showlegend=False
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Download Option für Bar Chart Data
            with st.expander("📥 Bar Chart Daten herunterladen"):
                bar_data = pd.DataFrame({
                    'Feature': top_features_de,
                    'SHAP_Wert': top_shap_values
                })
                download_link = create_download_link(bar_data, f"shap_bar_data_index_{selected_index}.csv")
                st.markdown(download_link, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen des SHAP Bar Charts: {str(e)}")

    with tab3:
        # SHAP Waterfall Plot mit matplotlib
        try:
            shap_explanation = shap.Explanation(
                values=shap_values_test[selected_index],
                base_values=expected_value,
                data=x_test_original.iloc[selected_index].values,
                feature_names=feature_names
            )
            
            # Erstelle Plot
            fig_shap, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(shap_explanation, max_display=10, show=False)
            
            plt.title(f'SHAP Waterfall Plot - Index: {selected_index}\n'
                      f'Modelloutput: {predicted_probability:.3f} | Tatsächlich: {"Bad Risk" if actual_class == 1 else "Good Risk"}', 
                      fontweight='bold', fontsize=12, pad=20)
            
            st.pyplot(fig_shap, bbox_inches='tight')
            
            # Download Option
            with st.expander("📥 Waterfall Plot herunterladen"):
                download_link = create_download_link(fig_shap, f"waterfall_plot_index_{selected_index}.png", "png")
                st.markdown(download_link, unsafe_allow_html=True)
            
            plt.close(fig_shap)
            
        except Exception as e:
            st.error(f"Fehler beim Erstellen des SHAP Waterfall Plots: {str(e)}")

    # SHAP-Analyse Details
    st.subheader("🔬 Detaillierte SHAP-Analyse")

    # Berechne SHAP-Beiträge für ausgewählten Index
    feature_values = x_test_original.iloc[selected_index].values
    shap_contributions = pd.DataFrame({
        'Feature': feature_names,
        'Feature_DE': [feature_translations.get(f, f) for f in feature_names],
        'Value': feature_values,
        'SHAP_Contribution': shap_values_test[selected_index]
    })

    # Top-Features
    top_positive = shap_contributions[shap_contributions['SHAP_Contribution'] > 0].nlargest(3, 'SHAP_Contribution')
    top_negative = shap_contributions[shap_contributions['SHAP_Contribution'] < 0].nsmallest(3, 'SHAP_Contribution')

    # Zeige Top-Faktoren
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("🔺 Top 3 Faktoren die das Risiko erhöhen")
        if len(top_positive) > 0:
            for _, row in top_positive.iterrows():
                delta_value = f"+{row['SHAP_Contribution']:.4f}"
                st.metric(
                    label=row['Feature_DE'],
                    value=f"{row['Value']:.3f}",
                    delta=delta_value
                )
        else:
            st.info("Keine Faktoren erhöhen das Risiko signifikant.")

    with col4:
        st.subheader("🔻 Top 3 Faktoren die das Risiko verringern")
        if len(top_negative) > 0:
            for _, row in top_negative.iterrows():
                delta_value = f"{row['SHAP_Contribution']:.4f}"
                st.metric(
                    label=row['Feature_DE'],
                    value=f"{row['Value']:.3f}",
                    delta=delta_value
                )
        else:
            st.info("Keine Faktoren verringern das Risiko signifikant.")

    # Zusätzliche Analysen
    st.subheader("📈 Zusätzliche Analysen")

    # Batch-Analyse
    with st.expander("🔍 Batch-Analyse"):
        st.write("Analysieren Sie mehrere Datenpunkte gleichzeitig:")
        
        # Anzahl Datenpunkte für Batch-Analyse
        num_samples = st.slider("Anzahl zufälliger Datenpunkte:", 5, 50, 10)
        
        if st.button("🎲 Zufällige Batch-Analyse starten"):
            # Zufällige Auswahl von Datenpunkten
            random_indices = np.random.choice(len(y_test), size=num_samples, replace=False)
            
            # Erstelle Zusammenfassung
            batch_results = []
            for idx in random_indices:
                batch_results.append({
                    'Index': idx,
                    'Tatsächliche_Klasse': 'Bad Risk' if y_test.iloc[idx] == 1 else 'Good Risk',
                    'Vorhergesagte_Wahrscheinlichkeit': f"{y_pred_proba[idx]:.4f}",
                    'Modell_Entscheidung': 'Bad Risk' if y_pred[idx] == 1 else 'Good Risk',
                    'Status': 'Korrekt' if y_test.iloc[idx] == y_pred[idx] else 'Falsch',
                    'Top_Risiko_Faktor': feature_translations.get(
                        feature_names[np.argmax(shap_values_test[idx])], 
                        feature_names[np.argmax(shap_values_test[idx])]
                    ),
                    'Top_Schutz_Faktor': feature_translations.get(
                        feature_names[np.argmin(shap_values_test[idx])], 
                        feature_names[np.argmin(shap_values_test[idx])]
                    )
                })
            
            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df, use_container_width=True)
            
            # Statistiken
            accuracy = (batch_df['Status'] == 'Korrekt').mean()
            st.metric("Batch-Genauigkeit", f"{accuracy:.2%}")
            
            # Download Option
            download_link = create_download_link(batch_df, f"batch_analysis_{num_samples}_samples.csv")
            st.markdown(download_link, unsafe_allow_html=True)

    # Detaillierte SHAP Tabelle
    st.subheader("📋 Alle SHAP-Beiträge")
    
    # Sortiere nach absoluten SHAP-Werten
    shap_contributions_sorted = shap_contributions.reindex(
        shap_contributions['SHAP_Contribution'].abs().sort_values(ascending=False).index
    )

    # Erstelle formatierte Tabelle
    display_df = shap_contributions_sorted[['Feature_DE', 'Value', 'SHAP_Contribution']].copy()
    display_df.columns = ['Feature', 'Wert', 'SHAP Beitrag']
    display_df['Wert'] = display_df['Wert'].round(3)
    display_df['SHAP Beitrag'] = display_df['SHAP Beitrag'].round(4)

    # Stil für positive/negative Werte
    styled_df = display_df.style.applymap(highlight_shap, subset=['SHAP Beitrag'])
    st.dataframe(styled_df, use_container_width=True)

    # Download für komplette SHAP-Analyse
    st.subheader("📥 Download-Optionen")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        st.markdown("**SHAP-Daten**")
        download_link = create_download_link(display_df, f"shap_complete_analysis_index_{selected_index}.csv")
        st.markdown(download_link, unsafe_allow_html=True)
    
    with col_dl2:
        st.markdown("**Modell-Vorhersage**")
        prediction_data = pd.DataFrame({
            'Index': [selected_index],
            'Tatsächliche_Klasse': ['Bad Risk' if actual_class == 1 else 'Good Risk'],
            'Vorhergesagte_Wahrscheinlichkeit': [predicted_probability],
            'Modell_Entscheidung': ['Bad Risk' if predicted_class == 1 else 'Good Risk'],
            'Status': [status]
        })
        download_link = create_download_link(prediction_data, f"prediction_data_index_{selected_index}.csv")
        st.markdown(download_link, unsafe_allow_html=True)
    
    with col_dl3:
        st.markdown("**Top Faktoren**")
        top_factors_data = pd.concat([
            top_positive[['Feature_DE', 'Value', 'SHAP_Contribution']].assign(Typ='Risiko erhöhend'),
            top_negative[['Feature_DE', 'Value', 'SHAP_Contribution']].assign(Typ='Risiko verringernd')
        ])
        download_link = create_download_link(top_factors_data, f"top_factors_index_{selected_index}.csv")
        st.markdown(download_link, unsafe_allow_html=True)

    # Technische Details
    with st.expander("🔧 Technische Details"):
        st.subheader("Modell-Informationen")
        st.write(f"**Expected Value (Baseline):** {expected_value:.6f}")
        
        # Berechne Logit und Wahrscheinlichkeit
        logit_value = expected_value + np.sum(shap_values_test[selected_index])
        calculated_probability = 1 / (1 + np.exp(-logit_value))
        
        st.write(f"**SHAP Summe (Logit):** {logit_value:.6f}")
        st.write(f"**Berechnete Wahrscheinlichkeit:** {calculated_probability:.6f}")
        st.write(f"**Modell-Wahrscheinlichkeit:** {predicted_probability:.6f}")
        st.write(f"**Differenz:** {abs(calculated_probability - predicted_probability):.6f}")
        
        st.subheader("Feature-Statistiken")
        st.write(f"**Anzahl Features:** {len(feature_names)}")
        st.write(f"**Positive SHAP-Beiträge:** {len(shap_contributions[shap_contributions['SHAP_Contribution'] > 0])}")
        st.write(f"**Negative SHAP-Beiträge:** {len(shap_contributions[shap_contributions['SHAP_Contribution'] < 0])}")
        st.write(f"**Neutrale SHAP-Beiträge:** {len(shap_contributions[(shap_contributions['SHAP_Contribution'] < 0.001) & (shap_contributions['SHAP_Contribution'] > -0.001)])}")

if __name__ == "__main__":
    render_shap_analysis()