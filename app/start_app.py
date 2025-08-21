"""
Start-Script für die SHAP Streamlit App (Windows-kompatibel)
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Prüft ob alle Requirements installiert sind"""
    missing_packages = []
    required_packages = {
        'streamlit': 'streamlit',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'lightgbm': 'lightgbm',
        'shap': 'shap',
        'plotly': 'plotly',
        'scipy': 'scipy'
    }
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    return len(missing_packages) == 0, missing_packages

def install_requirements():
    """Installiert Requirements aus requirements.txt"""
    print("📦 Installiere benötigte Pakete...")
    try:
        # Versuche erst requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Installieren der Requirements: {e}")
        print("🔄 Versuche einzelne Pakete zu installieren...")
        
        # Fallback: Installiere Pakete einzeln
        packages = [
            "streamlit", "pandas", "numpy", "matplotlib", "seaborn", 
            "scikit-learn", "lightgbm", "shap", "plotly", "scipy"
        ]
        
        failed_packages = []
        for package in packages:
            try:
                print(f"   📦 Installiere {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError:
                failed_packages.append(package)
                print(f"   ❌ Fehler bei {package}")
        
        if failed_packages:
            print(f"\n⚠️  Folgende Pakete konnten nicht installiert werden: {', '.join(failed_packages)}")
            return False
        
        return True

def check_data_folder():
    """Prüft ob der Datenordner existiert"""
    data_folder = Path("../GiveMeSomeCredit")
    if not data_folder.exists():
        print("⚠️  Warnung: GiveMeSomeCredit Ordner nicht gefunden!")
        print("   Bitte laden Sie die Daten herunter und platzieren Sie sie im GiveMeSomeCredit/ Ordner")
        print("   Benötigte Dateien: cs-training.csv, cs-test.csv")
        print("")
        return False
    
    train_file = data_folder / "cs-training.csv"
    test_file = data_folder / "cs-test.csv"
    
    if not train_file.exists() or not test_file.exists():
        print("⚠️  Warnung: Datendateien nicht vollständig!")
        print(f"   Gefunden: cs-training.csv = {train_file.exists()}")
        print(f"   Gefunden: cs-test.csv = {test_file.exists()}")
        return False
    
    return True

def start_streamlit():
    """Startet die Streamlit App"""
    print("🌐 Starte Streamlit App...")
    print("   Browser öffnet sich automatisch unter: http://localhost:8501")
    print("   Zum Beenden: Ctrl+C im Terminal")
    print("")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    print("🚀 SHAP Explainable AI Streamlit App")
    print("=" * 50)
    
    # Prüfe Requirements
    requirements_ok, missing_packages = check_requirements()
    
    if not requirements_ok:
        print(f"📦 Fehlende Pakete gefunden: {', '.join(missing_packages)}")
        print("🔄 Starte Installation...")
        
        if not install_requirements():
            print("\n❌ Installation fehlgeschlagen!")
            print("💡 Manuelle Installation empfohlen:")
            print("   pip install streamlit pandas numpy matplotlib seaborn scikit-learn lightgbm shap plotly scipy")
            
            # Frage ob trotzdem fortfahren
            choice = input("\nTrotzdem fortfahren? (j/n): ").lower()
            if choice != 'j' and choice != 'ja':
                print("👋 Installation abgebrochen.")
                sys.exit(1)
        else:
            print("✅ Pakete erfolgreich installiert!")
    
    # Prüfe Daten
    data_available = check_data_folder()
    if not data_available:
        print("\n⚠️  Die App kann ohne Daten nicht vollständig funktionieren.")
        choice = input("Trotzdem fortfahren? (j/n): ").lower()
        if choice != 'j' and choice != 'ja':
            print("👋 App-Start abgebrochen.")
            sys.exit(1)
    
    # Starte App
    try:
        start_streamlit()
    except KeyboardInterrupt:
        print("\n👋 App beendet. Auf Wiedersehen!")
    except Exception as e:
        print(f"\n❌ Fehler beim Starten der App: {e}")
        print("\n💡 Mögliche Lösungen:")
        print("   - Stellen Sie sicher, dass alle Pakete installiert sind")
        print("   - Prüfen Sie ob streamlit korrekt installiert ist: pip show streamlit")
        print("   - Versuchen Sie: streamlit run streamlit_app.py")
        input("Drücken Sie Enter zum Beenden...")
