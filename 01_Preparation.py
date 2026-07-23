import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gc
from datetime import datetime
import gdown

# ============================================
# IMPORTS POUR LES GRAPHIQUES
# ============================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# ============================================
# CONFIGURATION DE LA PAGE
# ============================================
st.set_page_config(
    page_title="🛡️ PaySim - Analyse d'anomalies unifiée",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style personnalisé
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; }
    .alert-critical { background-color: #ff6b6b; color: white; padding: 5px 10px; border-radius: 5px; }
    .alert-high { background-color: #ffa94d; color: white; padding: 5px 10px; border-radius: 5px; }
    .alert-moderate { background-color: #ffd93d; color: #333; padding: 5px 10px; border-radius: 5px; }
    .alert-normal { background-color: #6bcb77; color: white; padding: 5px 10px; border-radius: 5px; }
    .step-completed { background-color: #d4edda; padding: 10px; border-radius: 8px; border-left: 5px solid #28a745; }
    .step-active { background-color: #fff3cd; padding: 10px; border-radius: 8px; border-left: 5px solid #ffc107; }
    .step-pending { background-color: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 5px solid #6c757d; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ PaySim Haïti - Analyse d\'anomalies unifiée</p>', unsafe_allow_html=True)

# ============================================
# INITIALISATION DE L'ÉTAT DE SESSION
# ============================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_prepared' not in st.session_state:
    st.session_state.df_prepared = None
if 'df_trained' not in st.session_state:
    st.session_state.df_trained = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False

# ============================================
# DÉFINITIONS COMMUNES
# ============================================

CASH_OUT_BINS = [(20, 99), (100, 249), (250, 499), (500, 999), (1000, 1999),
                  (2000, 3999), (4000, 7999), (8000, 11999), (12000, 19999),
                  (20000, 39999), (40000, 59999), (60000, 74999), (75000, 100000)]
CASH_OUT_FEES = [6, 12, 15, 40, 65, 115, 185, 275, 380, 640, 1050, 1400, 1600]

CASH_IN_BINS = CASH_OUT_BINS
CASH_IN_FEES = [0] * 13

P2P_BINS = [(10, 99), (100, 249), (250, 499), (500, 999), (1000, 1999),
            (2000, 3999), (4000, 7999), (8000, 11999), (12000, 19999),
            (20000, 39999), (40000, 59999), (60000, 74999), (75000, 100000)]
P2P_FEES = [0, 0, 5, 10, 25, 35, 50, 60, 70, 75, 100, 120, 130]

TRANSACTIONNELLES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "type_encoded", "variationOrig", "variationDest", "ratio_amount_balance",
    "is_drained", "frais",
]
COMPORTEMENTALES = [
    "NbTransactionsHeure", "NbTransactionsJour", "MontantCumuleHeure",
    "MontantCumuleJour", "NbDestinatairesjour", "NbEmetteursJour",
    "MontantMoyenJour", "MontantMaxJour", "EcartTypeMontantsJour",
    "NombreTypesTransactions",
]
TEMPORELLES = [
    "jour", "heure", "TempsDepuisDerniereTransaction",
    "PremiereHeure", "DerniereHeure", "step_night",
]

FEATURE_SETS = {
    "M1": TRANSACTIONNELLES,
    "M2": TRANSACTIONNELLES + COMPORTEMENTALES,
    "M3": TRANSACTIONNELLES + COMPORTEMENTALES + TEMPORELLES,
}

NUM_COLS = TRANSACTIONNELLES + COMPORTEMENTALES + TEMPORELLES

VARS_UNIVARIEES = ["amount", "frais", "ratio_amount_balance",
                   "oldbalanceOrg", "newbalanceOrig", "variationOrig"]

SEVERITY_MAP = {
    3: {"label": "🔴 Critique", "color": "#ff6b6b", "class": "alert-critical"},
    2: {"label": "🟠 Élevée", "color": "#ffa94d", "class": "alert-high"},
    1: {"label": "🟡 Modérée", "color": "#ffd93d", "class": "alert-moderate"},
    0: {"label": "🟢 Normale", "color": "#6bcb77", "class": "alert-normal"},
}

# ============================================
# CLASSE COLUMNSELECTOR
# ============================================
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]

# ============================================
# FONCTIONS DE PRÉPARATION
# ============================================

def filter_type(df, exclude=("DEBIT",)):
    return df[~df["type"].isin(exclude)].copy()

def filter_amount_range(df, low=10, high=100_000):
    return df[df["amount"].between(low, high)].copy()

def compute_fees(df):
    df = df.copy()
    amount = df["amount"]
    ttype = df["type"]

    conditions, fees = [], []

    for (lo, hi), fee in zip(CASH_OUT_BINS, CASH_OUT_FEES):
        conditions.append((ttype == "CASH_OUT") & amount.between(lo, hi))
        fees.append(fee)

    for (lo, hi), fee in zip(CASH_IN_BINS, CASH_IN_FEES):
        conditions.append((ttype == "CASH_IN") & amount.between(lo, hi))
        fees.append(fee)

    for (lo, hi), fee in zip(P2P_BINS, P2P_FEES):
        conditions.append(ttype.isin(["TRANSFER", "PAYMENT"]) & amount.between(lo, hi))
        fees.append(fee)

    df["frais"] = np.select(conditions, fees, default=0)
    return df

def recalculate_balances(df):
    df = df.copy()
    df["newbalanceOrig"] = df["oldbalanceOrg"] - (df["amount"] + df["frais"])
    df["newbalanceDest"] = df["oldbalanceDest"] + df["amount"]
    return df

def filter_balance_sanity(df, cap=100_000):
    df = df.copy()
    is_dest_client = df["nameDest"].str[0] == "C"
    is_orig_client = df["nameOrig"].str[0] == "C"

    mask = df["newbalanceOrig"] > 0
    mask &= ~(is_dest_client & (df["newbalanceDest"] > cap))
    mask &= ~(is_orig_client & (df["newbalanceOrig"] > cap))
    mask &= ~(is_orig_client & (df["oldbalanceOrg"] > cap))
    return df[mask]

def add_temporal_features(df, night_start=22, night_end=5):
    df = df.copy()
    df["heure"] = (df["step"] - 1) % 24
    df["jour"] = ((df["step"] - 1) // 24) + 1
    df["step_night"] = ((df["heure"] >= night_start) | (df["heure"] <= night_end)).astype(int)
    return df

def flag_drained_accounts(df):
    df = df.copy()
    df["is_drained"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] < 10)).astype(int)
    return df

def add_ratio_variation(df):
    df = df.copy()
    df["ratio_amount_balance"] = np.where(df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0)
    df["variationOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["variationDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df

def add_behavioral_features(df):
    df = df.sort_values(["nameOrig", "step"]).copy()

    df["TempsDepuisDerniereTransaction"] = df.groupby("nameOrig")["step"].diff().fillna(0)

    df["NbTransactionsHeure"] = df.groupby(["nameOrig", "jour", "heure"])["step"].transform("count")
    df["NbTransactionsJour"] = df.groupby(["nameOrig", "jour"])["step"].transform("count")

    df["MontantCumuleHeure"] = df.groupby(["nameOrig", "jour", "heure"])["amount"].transform("sum")
    df["MontantCumuleJour"] = df.groupby(["nameOrig", "jour"])["amount"].transform("sum")

    nb_dest = df.groupby(["nameOrig", "jour"])["nameDest"].nunique().rename("NbDestinatairesjour")
    df = df.merge(nb_dest, on=["nameOrig", "jour"], how="left")

    nb_orig = df.groupby(["nameDest", "jour"])["nameOrig"].nunique().rename("NbEmetteursJour")
    df = df.merge(nb_orig, on=["nameDest", "jour"], how="left")

    df["MontantMoyenJour"] = df.groupby(["nameOrig", "jour"])["amount"].transform("mean")
    df["MontantMaxJour"] = df.groupby(["nameOrig", "jour"])["amount"].transform("max")
    df["EcartTypeMontantsJour"] = df.groupby(["nameOrig", "jour"])["amount"].transform("std").fillna(0)

    nb_types = df.groupby(["nameOrig", "jour"])["type"].nunique().rename("NombreTypesTransactions")
    df = df.merge(nb_types, on=["nameOrig", "jour"], how="left")

    premiere = df.groupby(["nameOrig", "jour"])["heure"].min().rename("PremiereHeure")
    derniere = df.groupby(["nameOrig", "jour"])["heure"].max().rename("DerniereHeure")
    df = df.merge(premiere, on=["nameOrig", "jour"], how="left")
    df = df.merge(derniere, on=["nameOrig", "jour"], how="left")

    return df

def encode_type(df):
    df = df.copy()
    df["type_encoded"] = df["type"].astype("category").cat.codes
    return df

def run_preparation_pipeline(df):
    df = filter_type(df, exclude=("DEBIT",))
    df = filter_amount_range(df, low=10, high=100_000)
    df = compute_fees(df)
    df = recalculate_balances(df)
    df = filter_balance_sanity(df, cap=100_000)
    df = add_temporal_features(df, night_start=22, night_end=5)
    df = flag_drained_accounts(df)
    df = add_ratio_variation(df)
    df = add_behavioral_features(df)
    df = encode_type(df)
    return df

# ============================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================

def build_anomaly_pipeline(columns, contamination=0.01, random_state=42):
    return Pipeline(steps=[
        ("selection", ColumnSelector(columns)),
        ("normalisation", RobustScaler()),
        ("acp", PCA(n_components=0.95)),
        ("isolation_forest", IsolationForest(
            contamination=contamination, 
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )),
    ])

# ============================================
# FONCTIONS D'EXPLICATION
# ============================================

def expliquer_anomalie(row, scaled_row, df_complet, seuil_percentile=95):
    explications = []
    signals = []
    stats_comparaison = {}
    
    deviations = scaled_row.abs().sort_values(ascending=False)
    top_vars = deviations.head(12)
    
    for var in top_vars.index:
        if var in df_complet.columns:
            valeur_actuelle = row[var]
            mediane = df_complet[var].median()
            q1 = df_complet[var].quantile(0.25)
            q3 = df_complet[var].quantile(0.75)
            iqr = q3 - q1 if q3 != q1 else 1
            
            if iqr == 0:
                iqr = 1
            
            try:
                percentile = (df_complet[var] <= valeur_actuelle).mean() * 100
            except:
                percentile = 50
            
            ecart_iqr = scaled_row[var] if var in scaled_row.index else 0
            
            if abs(ecart_iqr) >= 3:
                niveau = "🔴 Extrême"
            elif abs(ecart_iqr) >= 2:
                niveau = "🟠 Significatif"
            elif abs(ecart_iqr) >= 1:
                niveau = "🟡 Modéré"
            else:
                niveau = "🟢 Normal"
            
            if abs(ecart_iqr) > 1:
                direction = "supérieure" if ecart_iqr > 0 else "inférieure"
                explications.append({
                    'Variable': var,
                    'Valeur actuelle': valeur_actuelle,
                    'Médiane': mediane,
                    'Écart (IQR)': ecart_iqr,
                    'Percentile': percentile,
                    'Niveau': niveau,
                    'Interprétation': f"Valeur {direction} à {abs(ecart_iqr):.2f} IQR de la médiane"
                })
    
    if row.get('amount', 0) > df_complet['amount'].quantile(0.95):
        signals.append({
            'type': '💰',
            'message': f"Montant extrêmement élevé : {row['amount']:,.0f} FCFA (>95e percentile)"
        })
    elif row.get('amount', 0) > df_complet['amount'].quantile(0.75):
        signals.append({
            'type': '💰',
            'message': f"Montant élevé : {row['amount']:,.0f} FCFA (>75e percentile)"
        })
    
    ratio = row.get('ratio_amount_balance', 0)
    if ratio > 0.8:
        signals.append({
            'type': '📊',
            'message': f"Ratio montant/solde très élevé : {ratio:.1%}"
        })
    elif ratio > 0.5:
        signals.append({
            'type': '📊',
            'message': f"Ratio montant/solde élevé : {ratio:.1%}"
        })
    
    if row.get('is_drained', 0) == 1:
        signals.append({
            'type': '🏦',
            'message': "Compte vidé : Le compte émetteur a été quasiment vidé"
        })
    
    if row.get('step_night', 0) == 1:
        signals.append({
            'type': '🌙',
            'message': "Transaction nocturne (22h-5h)"
        })
    
    freq_heure = row.get('NbTransactionsHeure', 0)
    seuil_freq = df_complet['NbTransactionsHeure'].quantile(0.95)
    if freq_heure > seuil_freq:
        signals.append({
            'type': '⏱️',
            'message': f"Fréquence horaire très élevée : {freq_heure:.0f} transactions/heure"
        })
    
    if 'type' in row.index and 'type' in df_complet.columns:
        type_median = df_complet[df_complet['type'] == row['type']]['amount'].median()
        if type_median > 0:
            ratio_type = row['amount'] / type_median
            if ratio_type > 5:
                signals.append({
                    'type': '📈',
                    'message': f"Montant extrême pour ce type : {ratio_type:.1f}x la médiane du type {row['type']}"
                })
            elif ratio_type > 3:
                signals.append({
                    'type': '📈',
                    'message': f"Montant élevé pour ce type : {ratio_type:.1f}x la médiane du type {row['type']}"
                })
    
    variation = row.get('variationOrig', 0)
    if abs(variation) > df_complet['variationOrig'].quantile(0.95):
        signals.append({
            'type': '📉',
            'message': f"Variation de solde très importante : {variation:,.0f} FCFA"
        })
    
    if 'type' in row.index and 'type' in df_complet.columns:
        transactions_similaires = df_complet[
            (df_complet['type'] == row['type']) &
            (df_complet['NbModelesAnomalie'] == 0) &
            (abs(df_complet['heure'] - row['heure']) <= 2) if 'heure' in df_complet.columns else (df_complet['type'] == row['type'])
        ]
        stats_comparaison = {
            'nb_similaires': len(transactions_similaires),
            'montant_moyen_normal': transactions_similaires['amount'].mean() if len(transactions_similaires) > 0 else 0,
            'frais_moyen_normal': transactions_similaires['frais'].mean() if len(transactions_similaires) > 0 else 0,
        }
    
    return explications, signals, stats_comparaison

def generer_resume_anomalie(row):
    severity = SEVERITY_MAP[row['NbModelesAnomalie']]['label']
    resume = f"Transaction {severity} - "
    
    if 'type' in row.index:
        resume += f"Type: {row['type']}, "
    resume += f"Montant: {row['amount']:,.0f} FCFA"
    
    if row['NbModelesAnomalie'] == 3:
        resume += " 🔴 Détectée par les 3 modèles"
    elif row['NbModelesAnomalie'] >= 2:
        resume += " 🟠 Détectée par 2 modèles"
    elif row['NbModelesAnomalie'] >= 1:
        resume += " 🟡 Détectée par 1 modèle"
    
    return resume

# ============================================
# AFFICHAGE DES ÉTAPES
# ============================================

def afficher_progression():
    steps = [
        ("1. 📂 Chargement", "step_completed" if st.session_state.df_prepared is not None else "step_active" if st.session_state.step == 1 else "step_pending"),
        ("2. 🤖 Entraînement", "step_completed" if st.session_state.df_trained is not None else "step_active" if st.session_state.step == 2 else "step_pending"),
        ("3. 📊 Dashboard", "step_active" if st.session_state.step == 3 and st.session_state.df_trained is not None else "step_pending"),
    ]
    
    cols = st.columns(len(steps))
    for i, (label, status) in enumerate(steps):
        with cols[i]:
            if status == "step_completed":
                st.markdown(f'<div class="step-completed">✅ {label}</div>', unsafe_allow_html=True)
            elif status == "step_active":
                st.markdown(f'<div class="step-active">⏳ {label}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="step-pending">⏸️ {label}</div>', unsafe_allow_html=True)

# ============================================
# ÉTAPE 1 : CHARGEMENT + PRÉPARATION (FUSIONNÉES)
# ============================================

st.markdown("---")

if st.session_state.df_prepared is None:
    st.markdown("## 📂 Étape 1 : Chargement et préparation du fichier PaySim")
    
    DRIVE_FILE_ID = "1ddwlGLpzmim1dzXy1hVR35aBq9EKJXuA"
    DRIVE_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view?usp=sharing"
    DIRECT_DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}"
    
    st.info(f"📁 Fichier source : [PaySim CSV]({DRIVE_URL})")
    
    # Paramètres de chargement
    st.markdown("### ⚙️ Paramètres de chargement")
    
    MAX_ROWS = st.selectbox(
        "Nombre de lignes à charger",
        options=[10000, 25000, 50000, 100000, 200000, 500000],
        index=2,  # 50 000 par défaut
        help="Chargez moins de lignes pour éviter les problèmes de mémoire sur Streamlit Cloud (limite ~1-2 GB)"
    )
    
    st.warning("""
    ⚠️ **Important :** Streamlit Cloud a une limite de mémoire d'environ 1-2 GB.
    - 50 000 lignes = recommandé (stable)
    - 100 000 lignes = risque modéré
    - 200 000+ lignes = risque élevé de crash
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        **Comment ça fonctionne :**
        - Le fichier est téléchargé depuis Google Drive
        - Seulement le nombre de lignes sélectionné est chargé
        - La préparation est faite immédiatement
        - Le fichier original (494 MB) est supprimé après préparation
        - Seul le résultat préparé est conservé en mémoire
        """)
    
    with col2:
        if st.button("🚀 Charger et préparer", type="primary", use_container_width=True):
            with st.spinner("Téléchargement et préparation des données..."):
                try:
                    # Téléchargement
                    output = "paysim_data.csv"
                    st.info("📥 Téléchargement depuis Google Drive...")
                    gdown.download(DIRECT_DOWNLOAD_URL, output, quiet=False)
                    
                    # Lecture avec limitation
                    st.info(f"📊 Lecture de {MAX_ROWS:,} lignes...")
                    
                    try:
                        df = pd.read_csv(output, encoding="utf-8", nrows=MAX_ROWS)
                    except UnicodeDecodeError:
                        df = pd.read_csv(output, encoding="latin-1", nrows=MAX_ROWS)
                    
                    st.info(f"✅ {len(df):,} lignes chargées")
                    
                    # Nettoyage
                    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
                    
                    # Vérification des colonnes
                    required_cols = {"type", "amount", "step", "nameOrig", "nameDest",
                                      "oldbalanceOrg", "oldbalanceDest"}
                    missing = required_cols - set(df.columns)
                    if missing:
                        st.error(f"❌ Colonnes manquantes : {sorted(missing)}")
                        st.stop()
                    
                    # PRÉPARATION IMMÉDIATE
                    st.info("🔧 Préparation des données...")
                    df = run_preparation_pipeline(df)
                    
                    # Optimisation mémoire
                    for col in df.select_dtypes(include=['float64']).columns:
                        df[col] = df[col].astype('float32')
                    for col in df.select_dtypes(include=['int64']).columns:
                        df[col] = df[col].astype('int32')
                    
                    # Nettoyage du fichier temporaire
                    if os.path.exists(output):
                        os.remove(output)
                        st.info("🗑️ Fichier CSV temporaire supprimé")
                    
                    # Libération de la mémoire
                    gc.collect()
                    
                    # Sauvegarde
                    st.session_state.df_prepared = df
                    st.session_state.step = 2
                    st.session_state.file_loaded = True
                    
                    memory_used = df.memory_usage(deep=True).sum() / 1024 / 1024
                    
                    st.success(f"""
                    ✅ Chargement et préparation terminés avec succès !
                    - 📊 {len(df):,} transactions préparées
                    - 💾 Mémoire utilisée : {memory_used:.2f} MB
                    - 🗑️ Fichier original (494 MB) libéré
                    """)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
                    st.markdown("""
                    **Solutions :**
                    1. Réduisez le nombre de lignes
                    2. Vérifiez votre connexion internet
                    3. Assurez-vous que le fichier est accessible
                    """)
    
    # Option alternative : upload manuel
    st.divider()
    st.markdown("**Ou téléchargez manuellement :**")
    uploaded_file = st.file_uploader(
        "Choisissez votre fichier CSV PaySim",
        type=['csv'],
        help="Le fichier doit être au format PaySim standard"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, nrows=MAX_ROWS)
            
            df.columns = df.columns.str.strip().str.replace('\ufeff', '')
            
            required_cols = {"type", "amount", "step", "nameOrig", "nameDest",
                              "oldbalanceOrg", "oldbalanceDest"}
            missing = required_cols - set(df.columns)
            if missing:
                st.error(f"❌ Colonnes manquantes : {sorted(missing)}")
                st.stop()
            
            df = run_preparation_pipeline(df)
            
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].astype('float32')
            for col in df.select_dtypes(include=['int64']).columns:
                df[col] = df[col].astype('int32')
            
            st.session_state.df_prepared = df
            st.session_state.step = 2
            st.session_state.file_loaded = True
            st.success(f"✅ {len(df):,} transactions chargées et préparées avec succès !")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# ============================================
# ÉTAPE 2 : ENTRAÎNEMENT DES MODÈLES
# ============================================

elif st.session_state.df_trained is None:
    afficher_progression()
    st.markdown("---")
    st.markdown("## 🤖 Étape 2 : Entraînement des modèles")
    
    df = st.session_state.df_prepared
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Lignes préparées", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", f"{len(df.columns)}")
    with col3:
        st.metric("💰 Frais total", f"{df['frais'].sum():,.0f} FCFA")
    
    st.markdown("### 🧬 Familles de variables")
    
    col_fam1, col_fam2, col_fam3 = st.columns(3)
    with col_fam1:
        st.markdown(f"**M1 — Transactionnelles** ({len(FEATURE_SETS['M1'])} variables)")
    with col_fam2:
        st.markdown(f"**M2 — + Comportementales** ({len(FEATURE_SETS['M2'])} variables)")
    with col_fam3:
        st.markdown(f"**M3 — + Temporelles** ({len(FEATURE_SETS['M3'])} variables)")
    
    st.markdown("### ⚙️ Paramètres d'entraînement")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        contamination = st.slider(
            "Contamination",
            min_value=0.001,
            max_value=0.10,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
    with col_param2:
        n_estimators = st.slider(
            "Nombre d'arbres",
            min_value=50,
            max_value=300,
            value=100,
            step=10
        )
    with col_param3:
        random_state = st.number_input(
            "random_state",
            min_value=0,
            value=42,
            step=1
        )
    
    if st.button("🚀 Lancer l'entraînement des 3 modèles", type="primary", use_container_width=True):
        with st.spinner("Entraînement en cours..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            scaler_m3 = None
            
            for i, (name, cols) in enumerate(FEATURE_SETS.items()):
                status_text.info(f"🌲 Entraînement du modèle {name}...")
                
                pipeline = build_anomaly_pipeline(
                    cols,
                    contamination=contamination,
                    random_state=random_state
                )
                pipeline.named_steps['isolation_forest'].n_estimators = n_estimators
                
                pipeline.fit(df)
                
                if name == "M3":
                    scaler_m3 = pipeline.named_steps['normalisation']
                
                df[f"Score_{name}"] = pipeline.decision_function(df)
                df[f"Prediction_{name}"] = pipeline.predict(df)
                
                n_anomalies = (df[f"Prediction_{name}"] == -1).sum()
                status_text.info(f"✅ {name} terminé : {n_anomalies:,} anomalies détectées ({n_anomalies/len(df)*100:.2f}%)")
                
                progress_bar.progress(int((i + 1) / len(FEATURE_SETS) * 100))
                gc.collect()
            
            df["NbModelesAnomalie"] = (
                (df["Prediction_M1"] == -1).astype(int) +
                (df["Prediction_M2"] == -1).astype(int) +
                (df["Prediction_M3"] == -1).astype(int)
            )
            
            df["Sévérité"] = df["NbModelesAnomalie"].map(lambda x: SEVERITY_MAP[x]['label'])
            
            st.session_state.df_trained = df
            st.session_state.scaler = scaler_m3
            st.session_state.step = 3
            progress_bar.progress(100)
            status_text.success("✅ Entraînement terminé avec succès !")
            
            st.rerun()

# ============================================
# ÉTAPE 3 : DASHBOARD
# ============================================

else:
    afficher_progression()
    st.markdown("---")
    st.markdown("## 📊 Étape 3 : Dashboard d'analyse")
    
    df = st.session_state.df_trained
    scaler = st.session_state.scaler
    
    if scaler is not None:
        X_scaled = scaler.transform(df[NUM_COLS])
    else:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df[NUM_COLS])
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=NUM_COLS, index=df.index)
    
    # KPIS
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    
    with k1:
        st.metric("📄 Transactions", f"{len(df):,}")
    with k2:
        st.metric("💰 Montant total", f"{df['amount'].sum():,.0f} FCFA")
    with k3:
        st.metric("🏷️ Frais totaux", f"{df['frais'].sum():,.0f} FCFA")
    with k4:
        n_critiques = (df['NbModelesAnomalie'] == 3).sum()
        st.metric("🔴 Alertes critiques", f"{n_critiques:,}")
    with k5:
        n_alertes = (df['NbModelesAnomalie'] >= 1).sum()
        st.metric("🚨 Signalées", f"{n_alertes:,}")
    with k6:
        taux_anomalies = (df['NbModelesAnomalie'] >= 1).sum() / len(df) * 100
        st.metric("📊 Taux d'anomalies", f"{taux_anomalies:.2f}%")
    
    st.divider()
    
    # ============================================
    # ONGLETS DU DASHBOARD
    # ============================================
    
    tab13, tab14, tab15, tab16, tab17, tab_anom = st.tabs([
        "📊 Univarié", "🔗 Corrélations & ACP", "🧩 Clustering",
        "🤝 Comparaison M1/M2/M3", "🌀 t-SNE", "🚨 Anomalies"
    ])
    
    # --------------------------------------------
    # 13. ANALYSE UNIVARIÉE
    # --------------------------------------------
    with tab13:
        st.markdown('<p class="sub-header">Distributions des variables clés</p>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, col in zip(axes.ravel(), VARS_UNIVARIEES):
            if col in df.columns:
                sns.histplot(df[col], bins=50, ax=ax, kde=True)
                ax.set_title(f"Distribution de {col}")
                ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('<p class="sub-header">Boxplots des variables clés</p>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, col in zip(axes.ravel(), VARS_UNIVARIEES):
            if col in df.columns:
                sns.boxplot(x=df[col], ax=ax)
                ax.set_title(f"Boxplot de {col}")
                ax.set_xlabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        if 'type' in df.columns:
            st.markdown('<p class="sub-header">Montant par type de transaction (échelle log)</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df, x="type", y="amount", ax=ax)
            ax.set_yscale("log")
            ax.set_title("Distribution du montant par type de transaction (échelle log)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # --------------------------------------------
    # 14. CORRÉLATIONS & ACP
    # --------------------------------------------
    with tab14:
        st.markdown('<p class="sub-header">Matrice de corrélation (Spearman)</p>', unsafe_allow_html=True)
        
        corr = df[NUM_COLS].corr(method="spearman")
        
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, 
                    cbar_kws={"shrink": 0.7}, ax=ax)
        ax.set_title("Matrice de corrélation (Spearman)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('<p class="sub-header">Corrélation avec Score_M3</p>', unsafe_allow_html=True)
        
        if "Score_M3" in df.columns:
            corr_score = (
                df[NUM_COLS + ["Score_M3"]]
                .corr(method="spearman")["Score_M3"]
                .drop("Score_M3")
                .sort_values()
            )
            
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = np.where(corr_score > 0, "steelblue", "indianred")
            corr_score.plot(kind="barh", color=colors, ax=ax)
            ax.set_title("Corrélation (Spearman) de chaque variable avec Score_M3")
            ax.set_xlabel("Coefficient de corrélation")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown('<p class="sub-header">Projection ACP 2D</p>', unsafe_allow_html=True)
        
        pca_2d = PCA(n_components=2, random_state=42)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        fig, ax = plt.subplots(figsize=(9, 7))
        scatter = ax.scatter(
            X_pca_2d[:, 0], X_pca_2d[:, 1],
            c=df["Prediction_M3"], cmap="coolwarm", alpha=0.6, s=15,
        )
        ax.set_xlabel(f"Axe 1 ({pca_2d.explained_variance_ratio_[0]:.1%} de variance)")
        ax.set_ylabel(f"Axe 2 ({pca_2d.explained_variance_ratio_[1]:.1%} de variance)")
        ax.set_title("Projection ACP 2D — couleur = Prediction_M3")
        fig.colorbar(scatter, ax=ax, label="Prediction_M3")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown('<p class="sub-header">Cercle des corrélations</p>', unsafe_allow_html=True)
        
        loadings = pd.DataFrame(
            pca_2d.components_.T, 
            index=NUM_COLS, 
            columns=["Axe 1", "Axe 2"]
        )
        
        fig, ax = plt.subplots(figsize=(7, 7))
        circle = plt.Circle((0, 0), 1, fill=False, color="grey", linestyle="--")
        ax.add_patch(circle)
        
        for var in loadings.index:
            x, y = loadings.loc[var]
            ax.arrow(0, 0, x, y, head_width=0.02, color="steelblue")
            ax.text(x * 1.1, y * 1.1, var, fontsize=8)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title("Cercle des corrélations (ACP à 2 axes)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # --------------------------------------------
    # 15. CLUSTERING
    # --------------------------------------------
    with tab15:
        st.markdown('<p class="sub-header">Recherche du nombre de clusters (Elbow / Silhouette)</p>', unsafe_allow_html=True)
        
        sample_size = st.slider(
            "Taille de l'échantillon",
            1000, min(50_000, len(df)), min(20_000, len(df)), 1000,
            key="cluster_sample_size"
        )
        
        if st.button("🔍 Lancer l'exploration Elbow / Silhouette", use_container_width=True):
            with st.spinner("Calcul de l'inertie et du score de silhouette pour k = 2..7..."):
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
                X_sample = X_scaled[sample_idx]
                
                K_RANGE = range(2, 8)
                inertias, silhouettes = [], []
                
                for k in K_RANGE:
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = km.fit_predict(X_sample)
                    inertias.append(km.inertia_)
                    silhouettes.append(silhouette_score(X_sample, labels))
                
                st.session_state["cluster_sample_idx"] = sample_idx
                st.session_state["elbow_results"] = (list(K_RANGE), inertias, silhouettes)
        
        if "elbow_results" in st.session_state:
            K_RANGE, inertias, silhouettes = st.session_state["elbow_results"]
            
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))
            
            axes[0].plot(K_RANGE, inertias, marker="o")
            axes[0].set_title("Méthode du coude")
            axes[0].set_xlabel("Nombre de clusters (k)")
            axes[0].set_ylabel("Inertie")
            
            axes[1].plot(K_RANGE, silhouettes, marker="o", color="darkorange")
            axes[1].set_title("Score de silhouette")
            axes[1].set_xlabel("Nombre de clusters (k)")
            axes[1].set_ylabel("Silhouette")
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        st.divider()
        st.markdown('<p class="sub-header">KMeans final</p>', unsafe_allow_html=True)
        
        n_clusters = st.number_input("Nombre de clusters", min_value=2, max_value=20, value=4)
        
        if st.button("🎯 Entraîner KMeans sur l'ensemble des données", use_container_width=True):
            with st.spinner(f"Entraînement de KMeans (k={n_clusters}) sur {len(df):,} lignes..."):
                kmeans_final = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
                df["Cluster_KMeans"] = kmeans_final.fit_predict(X_scaled)
                st.session_state["df_with_clusters"] = df.copy()
                st.session_state.clustering_done = True
        
        if st.session_state.clustering_done and "df_with_clusters" in st.session_state:
            df_c = st.session_state["df_with_clusters"]
            
            st.write("**Répartition des clusters KMeans :**")
            st.dataframe(
                df_c["Cluster_KMeans"].value_counts().sort_index().rename("Nb transactions"),
                use_container_width=True
            )
            
            st.write("**Croisement clusters × prédictions Isolation Forest (M3) :**")
            st.dataframe(
                pd.crosstab(df_c["Cluster_KMeans"], df_c["Prediction_M3"]),
                use_container_width=True
            )
        
        st.divider()
        st.markdown('<p class="sub-header">DBSCAN (sur échantillon)</p>', unsafe_allow_html=True)
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            eps = st.slider("eps", 0.1, 5.0, 1.5, 0.1)
        with col_db2:
            min_samples = st.slider("min_samples", 2, 50, 10, 1)
        
        if st.button("🔍 Lancer DBSCAN", use_container_width=True):
            with st.spinner("Clustering DBSCAN sur l'échantillon..."):
                if "cluster_sample_idx" not in st.session_state:
                    rng = np.random.default_rng(42)
                    sample_idx = rng.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
                    st.session_state["cluster_sample_idx"] = sample_idx
                
                sample_idx = st.session_state["cluster_sample_idx"]
                X_sample = X_scaled[sample_idx]
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels_dbscan_sample = dbscan.fit_predict(X_sample)
                
                n_clusters_dbscan = len(set(labels_dbscan_sample)) - (1 if -1 in labels_dbscan_sample else 0)
                n_bruit = int((labels_dbscan_sample == -1).sum())
                
                st.write(f"**{n_clusters_dbscan} clusters**, **{n_bruit} points de bruit** "
                         f"({n_bruit / len(X_sample):.1%} de l'échantillon)")
                
                df_sample = df.iloc[sample_idx].copy()
                df_sample["Cluster_DBSCAN"] = labels_dbscan_sample
                
                st.write("**Croisement bruit DBSCAN × prédiction Isolation Forest (M3) :**")
                crosstab = pd.crosstab(
                    df_sample["Cluster_DBSCAN"] == -1, df_sample["Prediction_M3"],
                    rownames=["Bruit DBSCAN"], colnames=["Prediction_M3"],
                )
                st.dataframe(crosstab, use_container_width=True)
    
    # --------------------------------------------
    # 16. COMPARAISON M1/M2/M3
    # --------------------------------------------
    with tab16:
        st.markdown('<p class="sub-header">Indice de Jaccard entre les ensembles d\'anomalies</p>', unsafe_allow_html=True)
        
        def jaccard_anomalies(pred_a, pred_b):
            a = set(df.index[pred_a == -1])
            b = set(df.index[pred_b == -1])
            if not a and not b:
                return 1.0
            return len(a & b) / len(a | b)
        
        jrows = []
        for m1, m2 in [("M1", "M2"), ("M1", "M3"), ("M2", "M3")]:
            j = jaccard_anomalies(df[f"Prediction_{m1}"], df[f"Prediction_{m2}"])
            jrows.append({
                "Paire": f"{m1} vs {m2}",
                "Indice de Jaccard": f"{j:.2%}",
                "Intersection": f"{len(set(df.index[df[f'Prediction_{m1}'] == -1]) & set(df.index[df[f'Prediction_{m2}'] == -1])):,}",
                "Union": f"{len(set(df.index[df[f'Prediction_{m1}'] == -1]) | set(df.index[df[f'Prediction_{m2}'] == -1])):,}"
            })
        
        st.dataframe(pd.DataFrame(jrows), use_container_width=True)
        
        st.markdown('<p class="sub-header">Répartition selon le nombre de modèles</p>', unsafe_allow_html=True)
        
        dist = df["NbModelesAnomalie"].value_counts().sort_index()
        st.bar_chart(dist)
        st.dataframe(dist.rename("Nb transactions"), use_container_width=True)
        
        st.markdown('<p class="sub-header">Matrice de confusion entre modèles</p>', unsafe_allow_html=True)
        
        if all(f"Prediction_{m}" in df.columns for m in ["M1", "M2", "M3"]):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            for idx, (m1, m2) in enumerate([("M1", "M2"), ("M1", "M3"), ("M2", "M3")]):
                confusion = pd.crosstab(
                    df[f"Prediction_{m1}"],
                    df[f"Prediction_{m2}"],
                    rownames=[m1], colnames=[m2]
                )
                sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f"{m1} vs {m2}")
            
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # --------------------------------------------
    # 17. T-SNE
    # --------------------------------------------
    with tab17:
        st.markdown('<p class="sub-header">Projection t-SNE (sur échantillon)</p>', unsafe_allow_html=True)
        st.caption("Le t-SNE est coûteux en calcul : lancez-le explicitement sur un échantillon raisonnable.")
        
        col_tsne1, col_tsne2 = st.columns(2)
        with col_tsne1:
            tsne_sample_size = st.slider(
                "Taille de l'échantillon",
                500, min(20_000, len(df)), min(5_000, len(df)), 500,
                key="tsne_sample_size"
            )
        with col_tsne2:
            perplexity = st.slider("Perplexity", 5, 50, 30, 1)
        
        if st.button("🌀 Lancer le t-SNE", use_container_width=True):
            with st.spinner("Calcul de la projection t-SNE... (peut prendre du temps)"):
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(X_scaled), size=min(tsne_sample_size, len(X_scaled)), replace=False)
                X_sample = X_scaled[sample_idx]
                
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
                X_tsne = tsne.fit_transform(X_sample)
                
                fig, ax = plt.subplots(figsize=(9, 7))
                scatter = ax.scatter(
                    X_tsne[:, 0], X_tsne[:, 1],
                    c=df["Prediction_M3"].values[sample_idx],
                    cmap="coolwarm", alpha=0.6, s=15,
                )
                ax.set_title("Projection t-SNE — couleur = Prediction_M3")
                fig.colorbar(scatter, ax=ax, label="Prediction_M3")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
    
    # --------------------------------------------
    # 18. ANOMALIES — EXPLORATION DÉTAILLÉE
    # --------------------------------------------
    with tab_anom:
        st.markdown('<p class="sub-header">🚨 Exploration détaillée des transactions signalées</p>', unsafe_allow_html=True)
        
        st.info("💡 Cette section permet d'analyser en profondeur chaque transaction anormale.")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            min_severity = st.selectbox(
                "Sévérité minimale",
                [0, 1, 2, 3],
                index=1,
                format_func=lambda x: SEVERITY_MAP[x]['label']
            )
        
        with col_f2:
            if 'type' in df.columns:
                types_available = sorted(df["type"].unique())
                selected_types = st.multiselect(
                    "Filtrer par type",
                    types_available,
                    default=types_available
                )
            else:
                selected_types = []
        
        with col_f3:
            montant_min, montant_max = st.slider(
                "Filtre montant (FCFA)",
                min_value=0,
                max_value=int(df['amount'].max()),
                value=(0, int(df['amount'].max()))
            )
        
        filtered = df[
            (df['NbModelesAnomalie'] >= min_severity) &
            (df['amount'] >= montant_min) &
            (df['amount'] <= montant_max)
        ]
        
        if 'type' in df.columns and selected_types:
            filtered = filtered[filtered['type'].isin(selected_types)]
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.metric("📄 Total anomalies", f"{len(filtered):,}")
        with col_s2:
            st.metric("💰 Montant total", f"{filtered['amount'].sum():,.0f} FCFA")
        with col_s3:
            if len(filtered) > 0:
                st.metric("📊 Montant moyen", f"{filtered['amount'].mean():,.0f} FCFA")
            else:
                st.metric("📊 Montant moyen", "0 FCFA")
        with col_s4:
            st.metric("🔴 Critiques", f"{(filtered['NbModelesAnomalie'] == 3).sum()}")
        
        if len(filtered) > 0 and 'type' in filtered.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            filtered['type'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel("Type de transaction")
            ax.set_ylabel("Nombre d'anomalies")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
        
        st.markdown("#### 📋 Liste des transactions")
        
        display_cols = ['type', 'amount', 'heure', 'jour', 'frais', 'NbModelesAnomalie', 'Sévérité']
        display_cols = [c for c in display_cols if c in filtered.columns]
        
        filtered_display = filtered.copy()
        filtered_display['Sévérité'] = filtered_display['NbModelesAnomalie'].map(
            lambda x: SEVERITY_MAP[x]['label']
        )
        
        st.dataframe(
            filtered_display[display_cols],
            use_container_width=True,
            height=400
        )
        
        st.divider()
        st.markdown('<p class="sub-header">🔬 Analyse approfondie d\'une transaction</p>', unsafe_allow_html=True)
        
        if len(filtered) > 0:
            selected_idx = st.selectbox(
                "Choisissez une transaction à analyser",
                options=filtered.index.tolist(),
                format_func=lambda idx: f"#{idx} - {df.loc[idx, 'type'] if 'type' in df.columns else 'Transaction'} - {df.loc[idx, 'amount']:,.0f} FCFA"
            )
            
            if selected_idx is not None:
                row = df.loc[selected_idx]
                scaled_row = X_scaled_df.loc[selected_idx]
                
                st.markdown(f"#### 📌 Transaction #{selected_idx}")
                st.markdown(f"*{generer_resume_anomalie(row)}*")
                
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric("🏷️ Type", row['type'] if 'type' in row.index else "N/A")
                with col_m2:
                    st.metric("💰 Montant", f"{row['amount']:,.0f} FCFA")
                with col_m3:
                    severity = SEVERITY_MAP[row['NbModelesAnomalie']]['label']
                    st.metric("🚨 Sévérité", severity)
                with col_m4:
                    heure = int(row['heure']) if 'heure' in row.index else 0
                    jour = int(row['jour']) if 'jour' in row.index else 0
                    st.metric("🕐 Heure/Jour", f"{heure}h - Jour {jour}")
                
                st.markdown("#### 🤖 Scores des modèles")
                col_sc1, col_sc2, col_sc3 = st.columns(3)
                for idx_mod, (col, nom) in enumerate(zip([col_sc1, col_sc2, col_sc3], ['M1', 'M2', 'M3'])):
                    with col:
                        pred = row[f'Prediction_{nom}']
                        statut = "🔴 ANOMALIE" if pred == -1 else "🟢 NORMAL"
                        st.metric(
                            f"Modèle {nom}",
                            statut,
                            delta=f"Score: {row[f'Score_{nom}']:.4f}",
                            delta_color="inverse" if pred == -1 else "normal"
                        )
                
                st.markdown("#### 📊 Analyse des écarts statistiques")
                
                deviations = scaled_row.abs().sort_values(ascending=False)
                top_vars = deviations.head(12)
                
                analysis_data = []
                for var in top_vars.index:
                    if var in df.columns:
                        valeur_actuelle = row[var]
                        mediane = df[var].median()
                        q1 = df[var].quantile(0.25)
                        q3 = df[var].quantile(0.75)
                        ecart_iqr = scaled_row[var]
                        
                        try:
                            percentile = (df[var] <= valeur_actuelle).mean() * 100
                        except:
                            percentile = 50
                        
                        if abs(ecart_iqr) >= 3:
                            niveau = "🔴 Extrême"
                        elif abs(ecart_iqr) >= 2:
                            niveau = "🟠 Significatif"
                        elif abs(ecart_iqr) >= 1:
                            niveau = "🟡 Modéré"
                        else:
                            niveau = "🟢 Normal"
                        
                        analysis_data.append({
                            'Variable': var,
                            'Valeur actuelle': f"{valeur_actuelle:.2f}",
                            'Médiane': f"{mediane:.2f}",
                            'Écart (IQR)': f"{ecart_iqr:+.2f}",
                            'Percentile': f"{percentile:.1f}%",
                            'Niveau': niveau,
                            'Interprétation': f"Valeur {'supérieure' if ecart_iqr > 0 else 'inférieure'} à la médiane de {abs(ecart_iqr):.2f} IQR"
                        })
                
                st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ecarts = [float(d['Écart (IQR)']) for d in analysis_data[:10]]
                variables = [d['Variable'] for d in analysis_data[:10]]
                colors = ['red' if e > 0 else 'blue' for e in ecarts]
                ax.barh(variables, ecarts, color=colors)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
                ax.set_xlabel("Écart en unités IQR")
                ax.set_title("Top 10 des variables les plus atypiques")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown("#### 🚩 Signaux d'alerte détectés")
                explications, signals, stats_comp = expliquer_anomalie(row, scaled_row, df)
                
                if signals:
                    for signal in signals:
                        st.write(f"{signal['type']} **{signal['message']}**")
                else:
                    st.write("✅ Aucun signal d'alerte métier évident")
                
                st.markdown("#### 💾 Export de l'analyse")
                if analysis_data:
                    csv_analysis = pd.DataFrame(analysis_data).to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger l'analyse détaillée",
                        data=csv_analysis,
                        file_name=f"analyse_anomalie_{selected_idx}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        st.divider()
        st.markdown("#### 💾 Export des données filtrées")
        if len(filtered) > 0:
            csv_global = filtered.to_csv(index=False)
            st.download_button(
                "📥 Télécharger toutes les transactions filtrées",
                data=csv_global,
                file_name="anomalies_filtrees.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # ============================================
    # BOUTON DE TÉLÉCHARGEMENT FINAL
    # ============================================
    st.divider()
    st.markdown("### 💾 Télécharger le dataset complet")
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        csv_full = df.to_csv(index=False)
        st.download_button(
            "📥 Télécharger df_traite.csv",
            data=csv_full,
            file_name="df_traite.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_dl2:
        sample_size = min(1000, len(df))
        csv_sample = df.sample(n=sample_size, random_state=42).to_csv(index=False)
        st.download_button(
            f"📥 Télécharger échantillon ({sample_size:,} lignes)",
            data=csv_sample,
            file_name="df_traite_echantillon.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption("🛡️ PaySim Haïti - Application unifiée d'analyse d'anomalies")
