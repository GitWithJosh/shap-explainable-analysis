"""
SHAP Explainable AI - Mehrseitige Streamlit Anwendung
Hauptnavigation und App-Koordination
"""
import streamlit as st
import sys
from pathlib import Path

# Füge das pages Verzeichnis zum Python Path hinzu
pages_dir = Path(__file__).parent / "pages"
sys.path.append(str(pages_dir))

# Import der Seiten-Module
try:
    from shap_analysis import render_shap_analysis
    from documentation import render_documentation
except ImportError as e:
    st.error(f"Fehler beim Importieren der Seiten-Module: {e}")
    st.info("Stelle sicher, dass die pages/ Verzeichnis und Module existieren.")
    st.stop()

# Konfiguration der Streamlit-Seite
st.set_page_config(
    page_title="SHAP Explainable AI - Kreditrisiko-Analyse",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # SHAP Explainable AI Kreditrisiko-Analyse
        
        Diese Anwendung demonstriert Explainable AI (XAI) durch SHAP-Analysen 
        für Kreditrisikobewertungen. Entwickelt für PA2 - Duales Studium.
        
        **Features:**
        - Interaktive SHAP-Visualisierungen
        - Umfassende Dokumentation
        - Download-Optionen
        - Batch-Analysen
        
        **Technologie-Stack:**
        - Streamlit
        - SHAP
        - LightGBM
        - Plotly
        """
    }
)

def render_navigation():
    """Rendert die Hauptnavigation"""
    st.sidebar.markdown("# 🧠 SHAP Explainable AI")
    st.sidebar.markdown("---")
    
    # Hauptseitenauswahl
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🔍 SHAP Analyse", "📚 Dokumentation"],
        index=0
    )
    
    return page

def render_header(page_name):
    """Rendert den Haupt-Header"""
    st.markdown("""
    <div style="padding: 1rem 0; border-bottom: 2px solid #f0f0f0; margin-bottom: 2rem;">
        <h1 style="margin: 0; color: #1f77b4;">🧠 SHAP Explainable AI</h1>
        <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 1.1em;">
            Kreditrisiko-Analyse mit Machine Learning Erklärbarkeit
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Breadcrumb Navigation
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span style="color: #888;">📍 Sie sind hier:</span> 
        <span style="color: #1f77b4; font-weight: bold;">{page_name}</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Hauptfunktion der Anwendung"""
    
    # Navigation rendern
    current_page = render_navigation()
    
    # Header rendern
    render_header(current_page)
    
    # Seiten-spezifischen Inhalt rendern
    try:
        if current_page == "🔍 SHAP Analyse":
            render_shap_analysis()
        elif current_page == "📚 Dokumentation":
            render_documentation()
        else:
            st.error(f"Unbekannte Seite: {current_page}")
    
    except Exception as e:
        st.error(f"Fehler beim Laden der Seite '{current_page}': {str(e)}")
        st.exception(e)
        
        # Fallback-Information
        st.info("""
        **Mögliche Lösungen:**
        1. Stelle sicher, dass alle Dependencies installiert sind
        2. Überprüfe, ob model_components.pkl existiert
        3. Starte die Anwendung neu
        """)
    
    # Footer
    render_footer()

def render_footer():
    """Rendert den Footer"""
    st.markdown("---")
    
    # Footer in drei Spalten
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎓 Über dieses Projekt**
        - Entwickelt für PA2
        - Duales Studium
        - Explainable AI Demo
        """)
    
    with col2:
        st.markdown("""
        **🛠️ Technologie**
        - Python + Streamlit
        - SHAP + LightGBM
        - Plotly Visualisierungen
        """)
    
    with col3:
        st.markdown("""
        **📊 Datensatz**
        - Give Me Some Credit
        - Kaggle Competition
        - ~150k Kreditdaten
        """)
    
    # Copyright
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; color: #888; font-size: 0.9em;">
        📈 SHAP Explainable AI Kreditrisiko-Analyse | Entwickelt mit Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()