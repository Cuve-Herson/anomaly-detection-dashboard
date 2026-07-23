"""
app_unifiee.py
Application PaySim complète - Un seul fichier pour tout le pipeline :
1. Chargement depuis Google Drive
2. Préparation des données
3. Entraînement des modèles
4. Dashboard d'analyse complet
"""

# ============================================
# ⚠️ st.set_page_config DOIT être la première commande
# ============================================
import streamlit as st
st.set_page_config(
    page_title="🛡️ PaySim - Détection d'anomalies",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import os
import gdown
import time
import joblib
from datetime import datetime
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
# CONFIGURATION
# ============================================

FILE_ID = "1ddwlGLpzmim1dzXy1hVR35aBq9EKJXuA"  # ID de votre fichier sur Google Drive
DOWNLOAD_PATH = "paysim_data.csv"
DATASET_PREPARE_PATH = "data/dataset_prepare.csv"
DF_TRAITE_PATH = "data/df_traite.csv"
MODELS_DIR = "models"

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #2c3e50; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; }
    .alert-critical { background-color: #ff6b6b; color: white; padding: 5px 10px; border-radius: 5px; }
    .alert-high { background-color: #ffa94d; color: white; padding: 5px 10px; border-radius: 5px; }
    .alert-moderate { background-color: #ffd93d; color: #333; padding: 5px 10px; border-radius: 5px; }
    .alert-normal { background-color: #6bcb77; color: white; padding: 5px 10px; border-radius: 5px; }
    .step-completed { color: #6bcb77; font-weight: bold; }
    .step-pending { color: #ffa94d; font-weight: bold; }
    .step-error { color: #ff6b6b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================
# ÉTAT DE LA SESSION
# ============================================

if 'step' not in st.session_state:
    st.session_state.step = 0  # 0=chargement, 1=préparation, 2=entraînement, 3=dashboard
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'df_prepared' not in st.session_state:
    st.session_state.df_prepared = None
if 'df_traite' not in st.session_state:
    st.session_state.df_traite = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# ============================================
# FONCTIONS - PRÉPARATION (alignées sur le notebook)
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
# FONCTIONS - ENTRAÎNEMENT
# ============================================

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


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]


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
# FONCTIONS - DASHBOARD (explications)
# ============================================

SEVERITY_MAP = {
    3: {"label": "🔴 Critique", "color": "#ff6b6b", "class": "alert-critical"},
    2: {"label": "🟠 Élevée", "color": "#ffa94d", "class": "alert-high"},
    1: {"label": "🟡 Modérée", "color": "#ffd93d", "class": "alert-moderate"},
    0: {"label": "🟢 Normale", "color": "#6bcb77", "class": "alert-normal"},
}

VARS_UNIVARIEES = ["amount", "frais", "ratio_amount_balance",
                   "oldbalanceOrg", "newbalanceOrig", "variationOrig"]


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


def expliquer_anomalie(row, scaled_row, df_complet):
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
    
    # Signaux d'alerte
    if row.get('amount', 0) > df_complet['amount'].quantile(0.95):
        signals.append({'type': '💰', 'message': f"Montant extrêmement élevé : {row['amount']:,.0f} FCFA (>95e percentile)"})
    
    ratio = row.get('ratio_amount_balance', 0)
    if ratio > 0.8:
        signals.append({'type': '📊', 'message': f"Ratio montant/solde très élevé : {ratio:.1%}"})
    
    if row.get('is_drained', 0) == 1:
        signals.append({'type': '🏦', 'message': "Compte vidé : Le compte émetteur a été quasiment vidé"})
    
    if row.get('step_night', 0) == 1:
        signals.append({'type': '🌙', 'message': "Transaction nocturne (22h-5h)"})
    
    return explications, signals, stats_comparaison


# ============================================
# INTERFACE PRINCIPALE
# ============================================

st.markdown('<p class="main-header">🛡️ PaySim — Détection d\'anomalies financières</p>', unsafe_allow_html=True)

# Barre de progression des étapes
st.subheader("📋 Pipeline complet")

col_prog1, col_prog2, col_prog3, col_prog4 = st.columns(4)

steps_status = [
    ("📥 Chargement", st.session_state.step >= 0),
    ("🔧 Préparation", st.session_state.step >= 1),
    ("🤖 Entraînement", st.session_state.step >= 2),
    ("📊 Dashboard", st.session_state.step >= 3)
]

for idx, (label, done) in enumerate(steps_status):
    with [col_prog1, col_prog2, col_prog3, col_prog4][idx]:
        if done:
            st.markdown(f"✅ **{label}**")
        else:
            st.markdown(f"⏳ **{label}**")

st.divider()

# ============================================
# ÉTAPE 0 : CHARGEMENT DEPUIS GOOGLE DRIVE
# ============================================

if st.session_state.step == 0:
    st.markdown('<p class="sub-header">📥 Étape 1 : Chargement depuis Google Drive</p>', unsafe_allow_html=True)
    
    st.info(f"📁 Fichier à charger : `PS_20174392719_1491204439457_log.csv`")
    
    if st.button("📥 Charger et lancer tout le pipeline", type="primary", use_container_width=True):
        
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # --- Chargement ---
            progress_placeholder.progress(10, text="📥 Téléchargement du fichier depuis Google Drive...")
            
            if os.path.exists(DOWNLOAD_PATH):
                status_placeholder.info("📂 Fichier déjà téléchargé, utilisation du cache local...")
                df_raw = pd.read_csv(DOWNLOAD_PATH)
            else:
                url = f"https://drive.google.com/uc?id={FILE_ID}"
                gdown.download(url, DOWNLOAD_PATH, quiet=False)
                df_raw = pd.read_csv(DOWNLOAD_PATH)
            
            progress_placeholder.progress(25, text="✅ Fichier chargé !")
            status_placeholder.success(f"✅ Fichier chargé : {len(df_raw):,} lignes, {len(df_raw.columns)} colonnes")
            
            st.session_state.df_raw = df_raw
            
            # --- Préparation ---
            progress_placeholder.progress(30, text="🔧 Préparation des données...")
            status_placeholder.info("🔧 Exécution du pipeline de préparation...")
            
            df_prepared = run_preparation_pipeline(df_raw)
            
            os.makedirs("data", exist_ok=True)
            df_prepared.to_csv(DATASET_PREPARE_PATH, index=False)
            
            progress_placeholder.progress(55, text="✅ Préparation terminée !")
            status_placeholder.success(f"✅ Préparation terminée : {len(df_prepared):,} lignes, {len(df_prepared.columns)} colonnes")
            
            st.session_state.df_prepared = df_prepared
            
            # --- Entraînement ---
            progress_placeholder.progress(60, text="🤖 Entraînement des modèles...")
            status_placeholder.info("🤖 Entraînement des 3 modèles (M1, M2, M3)...")
            
            df_traite = df_prepared.copy()
            model_pipelines = {}
            scaler_m3 = None
            
            for i, (name, cols) in enumerate(FEATURE_SETS.items()):
                status_placeholder.info(f"🌲 Entraînement du modèle {name}... ({i+1}/3)")
                
                pipeline = build_anomaly_pipeline(cols, contamination=0.01, random_state=42)
                pipeline.fit(df_prepared)
                
                df_traite[f"Score_{name}"] = pipeline.decision_function(df_prepared)
                df_traite[f"Prediction_{name}"] = pipeline.predict(df_prepared)
                
                if name == "M3":
                    scaler_m3 = pipeline.named_steps['normalisation']
                
                model_pipelines[name] = pipeline
                
                progress_placeholder.progress(65 + i * 10, text=f"✅ Modèle {name} entraîné")
            
            # Sauvegarder les modèles
            os.makedirs(MODELS_DIR, exist_ok=True)
            for name, pipeline in model_pipelines.items():
                joblib.dump(pipeline, f"{MODELS_DIR}/pipeline_{name}.joblib")
            if scaler_m3:
                joblib.dump(scaler_m3, f"{MODELS_DIR}/scaler_M3.joblib")
            
            # Sauvegarder df_traite
            df_traite["NbModelesAnomalie"] = (
                (df_traite["Prediction_M1"] == -1).astype(int) +
                (df_traite["Prediction_M2"] == -1).astype(int) +
                (df_traite["Prediction_M3"] == -1).astype(int)
            )
            df_traite.to_csv(DF_TRAITE_PATH, index=False)
            
            progress_placeholder.progress(95, text="✅ Entraînement terminé !")
            status_placeholder.success(f"✅ Entraînement terminé : {len(df_traite):,} lignes, {len(df_traite.columns)} colonnes")
            
            st.session_state.df_traite = df_traite
            st.session_state.models = model_pipelines
            st.session_state.scaler = scaler_m3
            
            progress_placeholder.progress(100, text="✅ Pipeline COMPLET !")
            status_placeholder.success("🎉 Toutes les étapes sont terminées avec succès !")
            
            st.session_state.step = 3
            
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"❌ Erreur : {str(e)}")
            st.exception(e)

# ============================================
# ÉTAPE 1-2 : AFFICHAGE DE LA PROGRESSION
# ============================================

elif st.session_state.step == 1 or st.session_state.step == 2:
    st.info("⏳ Traitement en cours... Veuillez patienter.")
    st.progress(100)

# ============================================
# ÉTAPE 3 : DASHBOARD COMPLET
# ============================================

elif st.session_state.step == 3:
    
    df_traite = st.session_state.df_traite
    df = df_traite  # Alias pour le dashboard
    
    # Vérification des colonnes
    if "NbModelesAnomalie" not in df.columns:
        df["NbModelesAnomalie"] = (
            (df["Prediction_M1"] == -1).astype(int) +
            (df["Prediction_M2"] == -1).astype(int) +
            (df["Prediction_M3"] == -1).astype(int)
        )
    
    # Normalisation pour les explications
    if st.session_state.scaler is None:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(df[NUM_COLS])
    else:
        X_scaled = st.session_state.scaler.transform(df[NUM_COLS])
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=NUM_COLS, index=df.index)
    
    # Ajout de la colonne Sévérité
    df["Sévérité"] = df["NbModelesAnomalie"].map(lambda x: SEVERITY_MAP[x]['label'])
    
    # --- KPIS ---
    st.subheader("📊 Indicateurs clés")
    
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
        taux_anomalies = n_alertes / len(df) * 100
        st.metric("📊 Taux d'anomalies", f"{taux_anomalies:.2f}%")
    
    st.divider()
    
    # --- ONGLETS ---
    tab_univ, tab_corr, tab_cluster, tab_comp, tab_tsne, tab_anom = st.tabs([
        "📊 Univarié", "🔗 Corrélations & ACP", "🧩 Clustering",
        "🤝 Comparaison M1/M2/M3", "🌀 t-SNE", "🚨 Anomalies"
    ])
    
    # ========================
    # ONGLET 1 : UNIVARIÉ
    # ========================
    with tab_univ:
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
        
        if 'type' in df.columns:
            st.markdown('<p class="sub-header">Montant par type de transaction</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(data=df, x="type", y="amount", ax=ax)
            ax.set_yscale("log")
            ax.set_title("Distribution du montant par type de transaction (échelle log)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
    # ========================
    # ONGLET 2 : CORRÉLATIONS & ACP
    # ========================
    with tab_corr:
        st.markdown('<p class="sub-header">Matrice de corrélation (Spearman)</p>', unsafe_allow_html=True)
        
        corr = df[NUM_COLS].corr(method="spearman")
        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True, 
                    cbar_kws={"shrink": 0.7}, ax=ax)
        ax.set_title("Matrice de corrélation (Spearman)")
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
    
    # ========================
    # ONGLET 3 : CLUSTERING
    # ========================
    with tab_cluster:
        st.markdown('<p class="sub-header">Clustering KMeans</p>', unsafe_allow_html=True)
        
        n_clusters = st.number_input("Nombre de clusters", min_value=2, max_value=20, value=4, key="cluster_k")
        
        if st.button("🎯 Entraîner KMeans", use_container_width=True):
            with st.spinner(f"Entraînement de KMeans (k={n_clusters})..."):
                kmeans = KMeans(n_clusters=int(n_clusters), random_state=42, n_init=10)
                df["Cluster_KMeans"] = kmeans.fit_predict(X_scaled)
                st.success("✅ KMeans entraîné !")
                
                st.write("**Répartition des clusters :**")
                st.dataframe(df["Cluster_KMeans"].value_counts().sort_index().rename("Nb transactions"), use_container_width=True)
                
                st.write("**Croisement clusters × prédictions M3 :**")
                st.dataframe(pd.crosstab(df["Cluster_KMeans"], df["Prediction_M3"]), use_container_width=True)
        
        st.divider()
        st.markdown('<p class="sub-header">DBSCAN</p>', unsafe_allow_html=True)
        
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            eps = st.slider("eps", 0.1, 5.0, 1.5, 0.1, key="db_eps")
        with col_db2:
            min_samples = st.slider("min_samples", 2, 50, 10, 1, key="db_min")
        
        if st.button("🔍 Lancer DBSCAN", use_container_width=True):
            with st.spinner("Clustering DBSCAN..."):
                sample_size = min(20000, len(df))
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(X_scaled), size=sample_size, replace=False)
                X_sample = X_scaled[sample_idx]
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_sample)
                
                n_clusters_db = len(set(labels)) - (1 if -1 in labels else 0)
                n_bruit = int((labels == -1).sum())
                
                st.write(f"**{n_clusters_db} clusters**, **{n_bruit} points de bruit** ({n_bruit/len(X_sample):.1%})")
                
                df_sample = df.iloc[sample_idx].copy()
                df_sample["Cluster_DBSCAN"] = labels
                
                st.write("**Croisement bruit DBSCAN × Prediction_M3 :**")
                st.dataframe(pd.crosstab(df_sample["Cluster_DBSCAN"] == -1, df_sample["Prediction_M3"]), use_container_width=True)
    
    # ========================
    # ONGLET 4 : COMPARAISON M1/M2/M3
    # ========================
    with tab_comp:
        st.markdown('<p class="sub-header">Indice de Jaccard entre modèles</p>', unsafe_allow_html=True)
        
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
            })
        st.dataframe(pd.DataFrame(jrows), use_container_width=True)
        
        st.markdown('<p class="sub-header">Répartition selon le nombre de modèles</p>', unsafe_allow_html=True)
        dist = df["NbModelesAnomalie"].value_counts().sort_index()
        st.bar_chart(dist)
        
        st.markdown('<p class="sub-header">Matrice de confusion entre modèles</p>', unsafe_allow_html=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for idx, (m1, m2) in enumerate([("M1", "M2"), ("M1", "M3"), ("M2", "M3")]):
            confusion = pd.crosstab(df[f"Prediction_{m1}"], df[f"Prediction_{m2}"], rownames=[m1], colnames=[m2])
            sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f"{m1} vs {m2}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    # ========================
    # ONGLET 5 : T-SNE
    # ========================
    with tab_tsne:
        st.markdown('<p class="sub-header">Projection t-SNE</p>', unsafe_allow_html=True)
        
        col_tsne1, col_tsne2 = st.columns(2)
        with col_tsne1:
            tsne_size = st.slider("Taille échantillon", 500, 10000, 3000, 500, key="tsne_size")
        with col_tsne2:
            perplexity = st.slider("Perplexity", 5, 50, 30, 1, key="tsne_perp")
        
        if st.button("🌀 Lancer t-SNE", use_container_width=True):
            with st.spinner("Calcul du t-SNE... (peut prendre du temps)"):
                rng = np.random.default_rng(42)
                sample_idx = rng.choice(len(X_scaled), size=min(tsne_size, len(X_scaled)), replace=False)
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
    
    # ========================
    # ONGLET 6 : ANOMALIES
    # ========================
    with tab_anom:
        st.markdown('<p class="sub-header">🚨 Exploration détaillée des anomalies</p>', unsafe_allow_html=True)
        
        # Filtres
        st.markdown("#### 🔍 Filtres")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            min_severity = st.selectbox("Sévérité minimale", [0, 1, 2, 3], index=1,
                                        format_func=lambda x: SEVERITY_MAP[x]['label'], key="sev_filter")
        with col_f2:
            if 'type' in df.columns:
                types_available = sorted(df["type"].unique())
                selected_types = st.multiselect("Filtrer par type", types_available, default=types_available, key="type_filter")
            else:
                selected_types = []
        with col_f3:
            montant_min, montant_max = st.slider("Filtre montant (FCFA)", 0, int(df['amount'].max()), 
                                                 (0, int(df['amount'].max())), key="amt_filter")
        
        filtered = df[
            (df['NbModelesAnomalie'] >= min_severity) &
            (df['amount'] >= montant_min) &
            (df['amount'] <= montant_max)
        ]
        if 'type' in df.columns and selected_types:
            filtered = filtered[filtered['type'].isin(selected_types)]
        
        # Stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("📄 Total anomalies", f"{len(filtered):,}")
        with col_s2:
            st.metric("💰 Montant total", f"{filtered['amount'].sum():,.0f} FCFA")
        with col_s3:
            st.metric("📊 Montant moyen", f"{filtered['amount'].mean():,.0f} FCFA" if len(filtered) > 0 else "0 FCFA")
        with col_s4:
            st.metric("🔴 Critiques", f"{(filtered['NbModelesAnomalie'] == 3).sum()}")
        
        if len(filtered) > 0 and 'type' in filtered.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            filtered['type'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel("Type")
            ax.set_ylabel("Nombre d'anomalies")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
        
        # Liste
        st.markdown("#### 📋 Liste des transactions")
        display_cols = ['type', 'amount', 'heure', 'jour', 'frais', 'NbModelesAnomalie', 'Sévérité']
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, height=400)
        
        # Analyse détaillée
        st.divider()
        st.markdown('<p class="sub-header">🔬 Analyse approfondie</p>', unsafe_allow_html=True)
        
        if len(filtered) > 0:
            selected_idx = st.selectbox(
                "Choisissez une transaction à analyser",
                options=filtered.index.tolist(),
                format_func=lambda idx: f"#{idx} - {df.loc[idx, 'type'] if 'type' in df.columns else 'Transaction'} - {df.loc[idx, 'amount']:,.0f} FCFA",
                key="detail_select"
            )
            
            if selected_idx is not None:
                row = df.loc[selected_idx]
                scaled_row = X_scaled_df.loc[selected_idx]
                
                st.markdown(f"#### 📌 Transaction #{selected_idx}")
                st.markdown(f"*{generer_resume_anomalie(row)}*")
                
                # Métriques
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("🏷️ Type", row['type'] if 'type' in row.index else "N/A")
                with col_m2:
                    st.metric("💰 Montant", f"{row['amount']:,.0f} FCFA")
                with col_m3:
                    st.metric("🚨 Sévérité", SEVERITY_MAP[row['NbModelesAnomalie']]['label'])
                with col_m4:
                    heure = int(row['heure']) if 'heure' in row.index else 0
                    jour = int(row['jour']) if 'jour' in row.index else 0
                    st.metric("🕐 Heure/Jour", f"{heure}h - Jour {jour}")
                
                # Scores
                st.markdown("#### 🤖 Scores des modèles")
                col_sc1, col_sc2, col_sc3 = st.columns(3)
                for idx_mod, (col, nom) in enumerate(zip([col_sc1, col_sc2, col_sc3], ['M1', 'M2', 'M3'])):
                    with col:
                        pred = row[f'Prediction_{nom}']
                        statut = "🔴 ANOMALIE" if pred == -1 else "🟢 NORMAL"
                        st.metric(f"Modèle {nom}", statut, delta=f"Score: {row[f'Score_{nom}']:.4f}",
                                  delta_color="inverse" if pred == -1 else "normal")
                
                # Analyse des écarts
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
                        
                        if abs(ecart_iqr) > 1:
                            analysis_data.append({
                                'Variable': var,
                                'Valeur actuelle': f"{valeur_actuelle:.2f}",
                                'Médiane': f"{mediane:.2f}",
                                'Écart (IQR)': f"{ecart_iqr:+.2f}",
                                'Percentile': f"{percentile:.1f}%",
                                'Niveau': niveau,
                            })
                
                if analysis_data:
                    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
                    
                    # Visualisation des écarts
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
                
                # Signaux d'alerte
                st.markdown("#### 🚩 Signaux d'alerte détectés")
                explications, signals, _ = expliquer_anomalie(row, scaled_row, df)
                if signals:
                    for signal in signals:
                        st.write(f"{signal['type']} **{signal['message']}**")
                else:
                    st.write("✅ Aucun signal d'alerte métier évident")
        
        # Export
        st.divider()
        if len(filtered) > 0:
            csv_global = filtered.to_csv(index=False)
            st.download_button("📥 Télécharger toutes les transactions filtrées", data=csv_global,
                               file_name="anomalies_filtrees.csv", mime="text/csv", use_container_width=True)

else:
    st.info("🚀 Cliquez sur le bouton ci-dessus pour lancer l'ensemble du pipeline.")

st.divider()
st.caption("🛡️ PaySim - Détection d'anomalies financières (Pipeline unifié)")
