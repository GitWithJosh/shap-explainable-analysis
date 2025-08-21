# model_utils.py
# Hilfsmodule für die Streamlit-App

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy import stats, special
import shap
import pickle  # Verwende das eingebaute pickle statt pickle5
import os

def preprocess_data():
    """Laden und Vorverarbeitung der Daten wie im Notebook"""
    
    # Datenimport
    train_data = pd.read_csv('../GiveMeSomeCredit/cs-training.csv')
    test_data = pd.read_csv('../GiveMeSomeCredit/cs-test.csv')
    
    # Datenvorbereitung
    train_data_final = train_data.copy()
    test_data_final = test_data.copy()
    
    # Target-Variable aus Testdaten entfernen (ist ohnehin leer)
    if 'SeriousDlqin2yrs' in test_data_final.columns:
        test_data_final.drop('SeriousDlqin2yrs', axis=1, inplace=True)
    
    # IDs aus Datensätzen entfernen
    train_data_final.drop('Unnamed: 0', axis=1, inplace=True)
    test_data_final.drop('Unnamed: 0', axis=1, inplace=True)
    
    # Ersetze extreme Werte im Trainingsdatensatz
    train_data_final.loc[train_data_final['NumberOfTime30-59DaysPastDueNotWorse'] >= 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 13
    train_data_final.loc[train_data_final['NumberOfTime60-89DaysPastDueNotWorse'] >= 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 11
    train_data_final.loc[train_data_final['NumberOfTimes90DaysLate'] >= 90, 'NumberOfTimes90DaysLate'] = 17
    
    # Ersetze extreme Werte im Testdatensatz
    test_data_final.loc[test_data_final['NumberOfTime30-59DaysPastDueNotWorse'] >= 90, 'NumberOfTime30-59DaysPastDueNotWorse'] = 19
    test_data_final.loc[test_data_final['NumberOfTime60-89DaysPastDueNotWorse'] >= 90, 'NumberOfTime60-89DaysPastDueNotWorse'] = 9
    test_data_final.loc[test_data_final['NumberOfTimes90DaysLate'] >= 90, 'NumberOfTimes90DaysLate'] = 18
    
    # Entferne problematische Datensätze
    train_data_final = train_data_final[-((train_data_final['DebtRatio'] > train_data_final['DebtRatio'].quantile(0.95)) &
                                          (train_data_final['SeriousDlqin2yrs'] == train_data_final['MonthlyIncome']))]
    train_data_final = train_data_final[train_data_final['RevolvingUtilizationOfUnsecuredLines'] < 13]
    
    # Imputation fehlender Werte
    train_data_final['MonthlyIncome'].fillna(train_data_final['MonthlyIncome'].median(), inplace=True)
    train_data_final['NumberOfDependents'].fillna(0, inplace=True)
    test_data_final['MonthlyIncome'].fillna(test_data_final['MonthlyIncome'].median(), inplace=True)
    test_data_final['NumberOfDependents'].fillna(0, inplace=True)
    
    return train_data_final, test_data_final

def add_features(df):
    """Feature Engineering wie im Notebook"""
    df['MonthlyIncomePerPerson'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['MonthlyIncomePerPerson'].fillna(df['MonthlyIncomePerPerson'].median(), inplace=True)
    df['MonthlyDebt'] = df['DebtRatio'] * df['MonthlyIncome']
    df['MonthlyDebt'].fillna(df['MonthlyDebt'].median(), inplace=True)
    df['isRetired'] = (np.where(df['age'] >= 65, 1, 0)).astype(int)
    df['RevolvingLines'] = df['NumberOfOpenCreditLinesAndLoans'] - df['NumberRealEstateLoansOrLines']
    df['hasRevolvingLines'] = np.where(df['RevolvingLines'] > 0, 1, 0).astype(int)
    df['hasMultipleRealEstates'] = np.where(df['NumberRealEstateLoansOrLines'] > 1, 1, 0).astype(int)
    return df

def skew_measure(df):
    """Measure skewness of features"""
    nonObjectColList = df.dtypes[df.dtypes != 'object'].index
    skewM = df[nonObjectColList].apply(lambda x: stats.skew(x.dropna().sort_values(ascending=False)))
    skewM = pd.DataFrame({'skewness': skewM})
    return skewM[abs(skewM['skewness']) > 0.5].sort_values(by='skewness', ascending=False)

def apply_box_cox_transformation(train_data_final, test_data_final):
    """Apply Box-Cox transformation"""
    skewness_train = skew_measure(train_data_final)
    skewness_test = skew_measure(test_data_final)
    
    lambda_value = 0.15
    
    for feature in skewness_train.index:
        if feature != 'SeriousDlqin2yrs':
            train_data_final[feature] = special.boxcox1p(train_data_final[feature], lambda_value)
    
    for feature in skewness_test.index:
        if feature != 'SeriousDlqin2yrs':
            test_data_final[feature] = special.boxcox1p(test_data_final[feature], lambda_value)
    
    return train_data_final, test_data_final, skewness_train

def train_model():
    """Trainiert das LightGBM-Modell und speichert alle notwendigen Komponenten"""
    
    # Daten laden und vorverarbeiten
    train_data_final, test_data_final = preprocess_data()
    
    # Feature Engineering
    train_data_final = add_features(train_data_final)
    test_data_final = add_features(test_data_final)
    
    # Box-Cox Transformation
    train_data_final, test_data_final, skewness_train = apply_box_cox_transformation(train_data_final, test_data_final)
    
    # Train-Test Split
    x_train, x_test, y_train, y_test = train_test_split(
        train_data_final.drop('SeriousDlqin2yrs', axis=1), 
        train_data_final['SeriousDlqin2yrs'], 
        test_size=0.3, 
        random_state=42, 
        stratify=train_data_final['SeriousDlqin2yrs']
    )
    
    # LightGBM-Modell mit besten Parametern
    best_lgb_estimator = lgb.LGBMClassifier(
        objective='binary', 
        n_jobs=-1, 
        importance_type='gain', 
        random_state=42,
        subsample=0.9, 
        scale_pos_weight=20, 
        num_leaves=70, 
        n_estimators=400,
        min_split_gain=0.15, 
        min_data_in_leaf=300, 
        min_child_weight=7,
        max_depth=5, 
        learning_rate=0.05, 
        colsample_bytree=0.6
    )
    
    # Modell trainieren
    best_lgb_estimator.fit(x_train, y_train, feature_name=x_train.columns.tolist())
    
    # SHAP Explainer erstellen
    explainer = shap.TreeExplainer(best_lgb_estimator)
    shap_values_test = explainer.shap_values(x_test)
    
    # Rücktransformation für Interpretierbarkeit
    lambda_value = 0.15
    x_test_original = x_test.copy()
    
    for feature in skewness_train.index:
        if feature in x_test_original.columns:
            x_test_original[feature] = np.power(lambda_value * x_test_original[feature] + 1, 1/lambda_value) - 1
    
    # Vorhersagen
    y_pred_lgb = best_lgb_estimator.predict(x_test)
    y_pred_proba_lgb = best_lgb_estimator.predict_proba(x_test)[:, 1]
    
    # Alle Komponenten zurückgeben
    model_components = {
        'model': best_lgb_estimator,
        'explainer': explainer,
        'x_test': x_test,
        'x_test_original': x_test_original,
        'y_test': y_test,
        'y_pred': y_pred_lgb,
        'y_pred_proba': y_pred_proba_lgb,
        'shap_values_test': shap_values_test,
        'feature_names': list(x_test.columns),
        'expected_value': explainer.expected_value,
        'skewness_train': skewness_train
    }
    
    return model_components

def save_model_components(model_components, filepath='model_components.pkl'):
    """Speichert alle Modellkomponenten"""
    with open(filepath, 'wb') as f:
        pickle.dump(model_components, f)

def load_model_components(filepath='model_components.pkl'):
    """Lädt alle Modellkomponenten"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        print("Modellkomponenten nicht gefunden. Trainiere Modell...")
        components = train_model()
        save_model_components(components, filepath)
        return components

def get_feature_translations():
    """Deutsche Übersetzungen für Feature-Namen"""
    translations = {
        'RevolvingUtilizationOfUnsecuredLines': 'Kreditlinien-Auslastung',
        'age': 'Alter',
        'NumberOfTime30-59DaysPastDueNotWorse': 'Anzahl 30-59 Tage überfällig',
        'DebtRatio': 'Verschuldungsgrad',
        'MonthlyIncome': 'Monatseinkommen',
        'NumberOfOpenCreditLinesAndLoans': 'Anzahl offene Kredite/Kreditlinien',
        'NumberOfTimes90DaysLate': 'Anzahl 90+ Tage überfällig',
        'NumberRealEstateLoansOrLines': 'Anzahl Immobilienkredite',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Anzahl 60-89 Tage überfällig',
        'NumberOfDependents': 'Anzahl Angehörige',
        'MonthlyIncomePerPerson': 'Monatseinkommen pro Person',
        'MonthlyDebt': 'Monatliche Schulden',
        'isRetired': 'Rentner',
        'RevolvingLines': 'Revolvierende Kreditlinien',
        'hasRevolvingLines': 'Hat revolvierende Kreditlinien',
        'hasMultipleRealEstates': 'Hat mehrere Immobilien'
    }
    return translations
