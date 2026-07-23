"""
PY_03_Dashboard.py
Dashboard de présentation — version autonome (upload direct de df_traite.csv).

Reproduit les sections 13 à 17 du notebook PaySim_pipelines_avance.ipynb :
13. Analyse univariée
14. Analyse bi/multivariée (corrélations + ACP)
15. Clustering complémentaire (KMeans + DBSCAN)
16. Comparaison croisée M1/M2/M3
17. Visualisation t-SNE

+ un onglet "Anomalies" pour explorer/filtrer/télécharger les transactions
signalées avec explications détaillées.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# Configuration de la page
st.set_page_config(
    page_title="🛡️ Dashboard PaySim",
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🛡️ Dashboard d\'analyse des anomalies — PaySim</p>', unsafe_allow_html=True)

st.markdown("""
Ce dashboard reproduit les sections **13 à 17** du notebook
`PaySim_pipelines_avance.ipynb` : analyse univariée, corrélations/ACP,
clustering, comparaison des modèles M1/M2/M3, et t-SNE — plus un onglet
dédié à l'exploration détaillée des anomalies détectées.
""")

# ============================================
# DÉFINITIONS
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
NUM_COLS = TRANSACTIONNELLES + COMPORTEMENTALES + TEMPORELLES

VARS_UNIVARIEES = ["amount", "frais", "ratio_amount_balance",
                   "oldbalanceOrg", "newbalanceOrig", "variationOrig"]

# Dictionnaire de sévérité
SEVERITY_MAP = {
    3: {"label": "🔴 Critique", "color": "#ff6b6b", "class": "alert-critical"},
    2: {"label": "🟠 Élevée", "color": "#ffa94d", "class": "alert-high"},
    1: {"label": "🟡 Modérée", "color": "#ffd93d", "class": "alert-moderate"},
    0: {"label": "🟢 Normale", "color": "#6bcb77", "class": "alert-normal"},
}

# ============================================
# FONCTIONS D'EXPLICATION
# ============================================

def expliquer_anomalie(row, scaled_row, df_complet, seuil_percentile=95):
    """
    Génère une explication détaillée d'une anomalie avec contexte statistique.
    """
    explications = []
    alert_signals = []
    stats_comparaison = {}
    
    # 1. Analyse des écarts IQR
    deviations = scaled_row.abs().sort_values(ascending=False)
    top_vars = deviations.head(12)
    
    for var in top_vars.index:
        if var in df_complet.columns:
            valeur_actuelle = row[var]
            mediane = df_complet[var].median()
            q1 = df_complet[var].quantile(0.25)
            q3 = df_complet[var].quantile(0.75)
            iqr = q3 - q1 if q3 != q1 else 1
            
            # Éviter division par zéro
            if iqr == 0:
                iqr = 1
            
            # Calcul du percentile
            try:
                percentile = (df_complet[var] <= valeur_actuelle).mean() * 100
            except:
                percentile = 50
            
            ecart_iqr = scaled_row[var] if var in scaled_row.index else 0
            
            # Déterminer le niveau d'écart
            if abs(ecart_iqr) >= 3:
                niveau = "🔴 Extrême"
            elif abs(ecart_iqr) >= 2:
                niveau = "🟠 Significatif"
            elif abs(ecart_iqr) >= 1:
                niveau = "🟡 Modéré"
            else:
                niveau = "🟢 Normal"
            
            if abs(ecart_iqr) > 1:  # Ne garder que les écarts significatifs
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
    
    # 2. Règles métier pour les signaux d'alerte
    signals = []
    
    # Montant
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
    
    # Ratio amount/balance
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
    
    # Drainage
    if row.get('is_drained', 0) == 1:
        signals.append({
            'type': '🏦',
            'message': "Compte vidé : Le compte émetteur a été quasiment vidé"
        })
    
    # Transaction nocturne
    if row.get('step_night', 0) == 1:
        signals.append({
            'type': '🌙',
            'message': "Transaction nocturne (22h-5h)"
        })
    
    # Fréquence inhabituelle
    freq_heure = row.get('NbTransactionsHeure', 0)
    seuil_freq = df_complet['NbTransactionsHeure'].quantile(0.95)
    if freq_heure > seuil_freq:
        signals.append({
            'type': '⏱️',
            'message': f"Fréquence horaire très élevée : {freq_heure:.0f} transactions/heure"
        })
    
    # Type de transaction spécifique
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
    
    # Variation de solde
    variation = row.get('variationOrig', 0)
    if abs(variation) > df_complet['variationOrig'].quantile(0.95):
        signals.append({
            'type': '📉',
            'message': f"Variation de solde très importante : {variation:,.0f} FCFA"
        })
    
    # 3. Statistiques de comparaison
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
    """Génère un résumé textuel de l'anomalie."""
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
# CHARGEMENT DES DONNÉES
# ============================================

st.subheader("📂 Étape 1 : Chargement du dataset traité")

uploaded_file = st.file_uploader(
    "Choisissez votre fichier df_traite.csv (issu de PY_02_Training.py)",
    type=['csv'],
    help="Doit contenir les colonnes Score_M1/M2/M3 et Prediction_M1/M2/M3"
)

if uploaded_file is None:
    st.info("👆 Chargez votre fichier df_traite.csv pour afficher le dashboard")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, encoding="utf-8")
except UnicodeDecodeError:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, encoding="latin-1")
except Exception as e:
    st.error(f"❌ Impossible de lire le fichier CSV : {str(e)}")
    st.stop()

df.columns = df.columns.str.strip().str.replace('\ufeff', '')
st.success(f"✅ Fichier chargé : {uploaded_file.name} — {len(df):,} lignes, {len(df.columns)} colonnes")

# Vérification des colonnes requises
required = set(NUM_COLS) | {"Score_M1", "Score_M2", "Score_M3",
                            "Prediction_M1", "Prediction_M2", "Prediction_M3"}
missing = required - set(df.columns)
if missing:
    st.error(
        f"❌ Colonnes manquantes : {sorted(missing)}\n\n"
        "Vérifiez que ce fichier provient bien de PY_02_Training.py."
    )
    st.stop()

# ============================================
# NORMALISATION POUR L'EXPLICATION
# ============================================

# Essayer de charger le scaler sauvegardé
scaler = None
scaler_loaded = False

try:
    if os.path.exists("models/scaler_M3.joblib"):
        scaler = joblib.load("models/scaler_M3.joblib")
        scaler_loaded = True
        st.info("✅ Scaler M3 chargé depuis les modèles sauvegardés")
except Exception as e:
    st.warning(f"⚠️ Impossible de charger le scaler : {str(e)}")

# Si le scaler n'est pas disponible, réentraîner
if scaler is None:
    st.warning("⚠️ Scaler non trouvé - réentraînement sur les données actuelles")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[NUM_COLS])
else:
    X_scaled = scaler.transform(df[NUM_COLS])

X_scaled_df = pd.DataFrame(X_scaled, columns=NUM_COLS, index=df.index)

# ============================================
# KPIS
# ============================================

# Calcul des métriques
df["NbModelesAnomalie"] = (
    (df["Prediction_M1"] == -1).astype(int)
    + (df["Prediction_M2"] == -1).astype(int)
    + (df["Prediction_M3"] == -1).astype(int)
)

df["Sévérité"] = df["NbModelesAnomalie"].map(lambda x: SEVERITY_MAP[x]['label'])

# Affichage des KPIs
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
    taux_anomalies = (df['NbModelesAnomalie'] >= 1).sum() / len(df) * 100
    st.metric("📊 Taux d'anomalies", f"{taux_anomalies:.2f}%")

st.divider()

# ============================================
# ONGLETS PRINCIPAUX
# ============================================

tab13, tab14, tab15, tab16, tab17, tab_anom = st.tabs([
    "📊 13. Univarié", "🔗 14. Corrélations & ACP", "🧩 15. Clustering",
    "🤝 16. Comparaison M1/M2/M3", "🌀 17. t-SNE", "🚨 Anomalies"
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
    
    st.markdown('<p class="sub-header">Montant par type de transaction (échelle log)</p>', unsafe_allow_html=True)
    
    if 'type' in df.columns:
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
    
    # Calcul de la corrélation
    corr = df[NUM_COLS].corr(method="spearman")
    
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, 
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Matrice de corrélation (Spearman) des variables du modèle M3")
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
    
    # ACP
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
    ax.set_title("Cercle des corrélations (ACP à 2 axes, modèle M3)")
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
    
    if "df_with_clusters" in st.session_state and "Cluster_KMeans" in st.session_state["df_with_clusters"].columns:
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
    
    # Matrice de confusion entre modèles
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
# ANOMALIES — EXPLORATION DÉTAILLÉE
# --------------------------------------------
with tab_anom:
    st.markdown('<p class="sub-header">🚨 Exploration détaillée des transactions signalées</p>', unsafe_allow_html=True)
    
    st.info("💡 Cette section permet d'analyser en profondeur chaque transaction anormale pour comprendre pourquoi elle est considérée comme suspecte.")
    
    # Filtres
    st.markdown("#### 🔍 Filtres")
    
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
    
    # Application des filtres
    filtered = df[
        (df['NbModelesAnomalie'] >= min_severity) &
        (df['amount'] >= montant_min) &
        (df['amount'] <= montant_max)
    ]
    
    if 'type' in df.columns and selected_types:
        filtered = filtered[filtered['type'].isin(selected_types)]
    
    # Statistiques des anomalies
    st.markdown("#### 📊 Statistiques")
    
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    
    with col_s1:
        st.metric("📄 Total anomalies", f"{len(filtered):,}")
    
    with col_s2:
        total_montant = filtered['amount'].sum()
        st.metric("💰 Montant total", f"{total_montant:,.0f} FCFA")
    
    with col_s3:
        if len(filtered) > 0:
            montant_moyen = filtered['amount'].mean()
            st.metric("📊 Montant moyen", f"{montant_moyen:,.0f} FCFA")
        else:
            st.metric("📊 Montant moyen", "0 FCFA")
    
    with col_s4:
        alertes_critiques = (filtered['NbModelesAnomalie'] == 3).sum()
        st.metric("🔴 Critiques", f"{alertes_critiques}")
    
    if len(filtered) > 0 and 'type' in filtered.columns:
        st.markdown("#### 📊 Distribution par type")
        fig, ax = plt.subplots(figsize=(10, 4))
        filtered['type'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Type de transaction")
        ax.set_ylabel("Nombre d'anomalies")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)
    
    # Liste des transactions
    st.markdown("#### 📋 Liste des transactions")
    
    display_cols = ['type', 'amount', 'heure', 'jour', 'frais', 'NbModelesAnomalie', 'Sévérité']
    display_cols = [c for c in display_cols if c in filtered.columns]
    
    # Ajouter la sévérité formatée
    filtered_display = filtered.copy()
    filtered_display['Sévérité'] = filtered_display['NbModelesAnomalie'].map(
        lambda x: SEVERITY_MAP[x]['label']
    )
    
    st.dataframe(
        filtered_display[display_cols],
        use_container_width=True,
        height=400
    )
    
    # EXPLICATION DÉTAILLÉE D'UNE TRANSACTION
    st.divider()
    st.markdown('<p class="sub-header">🔬 Analyse approfondie d\'une transaction</p>', unsafe_allow_html=True)
    st.info("💡 Sélectionnez une transaction pour comprendre en détail pourquoi elle est considérée comme anormale")
    
    if len(filtered) > 0:
        # Sélection de la transaction
        selected_idx = st.selectbox(
            "Choisissez une transaction à analyser",
            options=filtered.index.tolist(),
            format_func=lambda idx: f"#{idx} - {df.loc[idx, 'type'] if 'type' in df.columns else 'Transaction'} - {df.loc[idx, 'amount']:,.0f} FCFA"
        )
        
        if selected_idx is not None:
            row = df.loc[selected_idx]
            scaled_row = X_scaled_df.loc[selected_idx]
            
            # Résumé de la transaction
            st.markdown(f"#### 📌 Transaction #{selected_idx}")
            st.markdown(f"*{generer_resume_anomalie(row)}*")
            
            # Métriques clés
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
            
            # Scores des modèles
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
            
            # Analyse détaillée des écarts
            st.markdown("#### 📊 Analyse des écarts statistiques")
            
            # Récupérer les écarts significatifs
            deviations = scaled_row.abs().sort_values(ascending=False)
            top_vars = deviations.head(12)
            
            # Créer un DataFrame pour l'affichage
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
                    
                    # Déterminer le niveau d'écart
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
            
            # Signaux d'alerte détectés
            st.markdown("#### 🚩 Signaux d'alerte détectés")
            
            # Générer les explications
            explications, signals, stats_comp = expliquer_anomalie(row, scaled_row, df)
            
            if signals:
                for signal in signals:
                    st.write(f"{signal['type']} **{signal['message']}**")
            else:
                st.write("✅ Aucun signal d'alerte métier évident - l'anomalie est détectée par une combinaison de facteurs")
            
            # Comparaison avec les transactions normales
            if stats_comp.get('nb_similaires', 0) > 0:
                st.markdown("#### 📊 Comparaison avec des transactions normales similaires")
                
                col_comp1, col_comp2, col_comp3 = st.columns(3)
                
                with col_comp1:
                    st.metric(
                        "📊 Transactions similaires normales",
                        f"{stats_comp['nb_similaires']}"
                    )
                
                with col_comp2:
                    if stats_comp['montant_moyen_normal'] > 0:
                        delta = row['amount'] - stats_comp['montant_moyen_normal']
                        st.metric(
                            "💰 Montant moyen normal",
                            f"{stats_comp['montant_moyen_normal']:,.0f} FCFA",
                            delta=f"{delta:+,.0f} FCFA",
                            delta_color="off" if delta > 0 else "normal"
                        )
                
                with col_comp3:
                    if stats_comp['frais_moyen_normal'] > 0:
                        delta = row['frais'] - stats_comp['frais_moyen_normal']
                        st.metric(
                            "💸 Frais moyens normaux",
                            f"{stats_comp['frais_moyen_normal']:,.0f} FCFA",
                            delta=f"{delta:+,.0f} FCFA",
                            delta_color="off" if delta > 0 else "normal"
                        )
            
            # Export de l'analyse
            st.markdown("#### 💾 Export de l'analyse")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                if analysis_data:
                    csv_analysis = pd.DataFrame(analysis_data).to_csv(index=False)
                    st.download_button(
                        "📥 Télécharger l'analyse détaillée",
                        data=csv_analysis,
                        file_name=f"analyse_anomalie_{selected_idx}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_exp2:
                # Détails complets de la transaction
                details = pd.DataFrame([row.to_dict()]).T
                details.columns = ['Valeur']
                csv_details = details.to_csv()
                st.download_button(
                    "📥 Télécharger les détails complets",
                    data=csv_details,
                    file_name=f"transaction_{selected_idx}_details.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        st.warning("Aucune transaction ne correspond aux filtres actuels.")
    
    # Export global
    st.divider()
    st.markdown("#### 💾 Export des données filtrées")
    
    if len(filtered) > 0:
        csv_global = filtered.to_csv(index=False)
        st.download_button(
            "📥 Télécharger toutes les transactions filtrées",
            data=csv_global,
            file_name="anomalies_filtrees_completes.csv",
            mime="text/csv",
            use_container_width=True
        )

st.divider()
st.caption("🛡️ Dashboard - Projet PaySim Haïti (sections 13 à 17 du notebook)")