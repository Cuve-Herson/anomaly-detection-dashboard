"""
PY_02_Training.py
Entraîne les modèles d'anomalie - Version autonome, alignée sur
PaySim_pipelines_avance.ipynb (sections 10 "Familles de variables",
11 "Pipelines de modélisation" et 12 "Comparaison des modèles").

Entrée attendue : le dataset_prepare.csv produit par PY_01_Preparation.py.
Sortie : df_traite.csv (dataset enrichi avec Score_M1/M2/M3 et
Prediction_M1/M2/M3), téléchargeable et utilisable directement dans
PY_03_Dashboard.py (sections 13 à 17 + anomalies).
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# IMPORT MANQUANTS - Ajout de matplotlib et seaborn
import matplotlib
matplotlib.use("Agg")  # Pour éviter les problèmes d'affichage
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Configuration de la page
st.set_page_config(
    page_title="🤖 Entraînement des modèles",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Entraînement des modèles d'anomalie PaySim")

st.markdown("""
Cette application reproduit la section **"Pipelines de modélisation"** du notebook :
1. **Charger** le `dataset_prepare.csv` produit par PY_01_Preparation.py
2. **Définir** les 3 familles de variables (M1, M2, M3)
3. **Construire** 3 pipelines `ColumnSelector → RobustScaler → PCA(0.95) → IsolationForest`
4. **Entraîner** chaque modèle et calculer scores + prédictions
5. **Sauvegarder** les modèles et le scaler pour réutilisation
6. **Comparer** les 3 modèles et **télécharger** le dataset enrichi
""")

# ============================================
# DÉFINITIONS — identiques à la section 10 du notebook
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


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Sélectionne un sous-ensemble de colonnes d'un DataFrame."""
    
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]


def build_anomaly_pipeline(columns, contamination=0.01, random_state=42):
    """Équivalent de build_anomaly_pipeline du notebook."""
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


def sauvegarder_modele(pipeline, nom_modele, scaler=None):
    """Sauvegarde le pipeline et le scaler pour une réutilisation ultérieure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    # Sauvegarder le pipeline complet
    chemin_pipeline = f"models/pipeline_{nom_modele}_{timestamp}.joblib"
    joblib.dump(pipeline, chemin_pipeline)
    
    # Si un scaler est fourni, le sauvegarder séparément (pour M3)
    if scaler is not None and nom_modele == "M3":
        chemin_scaler = "models/scaler_M3.joblib"
        joblib.dump(scaler, chemin_scaler)
        st.info(f"💾 Scaler M3 sauvegardé dans {chemin_scaler}")
    
    return chemin_pipeline


# ============================================
# 1. UPLOAD DU FICHIER PRÉPARÉ
# ============================================

st.subheader("📂 Étape 1 : Chargement du dataset préparé")

uploaded_file = st.file_uploader(
    "Choisissez votre fichier dataset_prepare.csv (issu de PY_01_Preparation.py)",
    type=['csv'],
    help="Doit contenir les colonnes générées par le pipeline de préparation du notebook"
)

if uploaded_file is not None:
    file_size = uploaded_file.size / 1024 / 1024
    st.success(f"✅ Fichier chargé : {uploaded_file.name} ({file_size:.2f} MB)")
    
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier CSV : {str(e)}")
        st.stop()
    
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    
    # Aperçu des données
    with st.expander("👁️ Aperçu des données chargées", expanded=False):
        st.write(f"**{len(df):,} lignes × {len(df.columns)} colonnes**")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Afficher les types de colonnes
        st.write("**Types de colonnes :**")
        dtype_df = pd.DataFrame({
            'Colonne': df.dtypes.index,
            'Type': df.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Vérification des colonnes requises
    all_required = set(FEATURE_SETS["M3"])
    missing = all_required - set(df.columns)
    
    if missing:
        st.error(
            f"❌ Colonnes manquantes pour reproduire les familles de variables : {sorted(missing)}\n\n"
            "Vérifiez que ce fichier provient bien de PY_01_Preparation.py."
        )
        st.stop()
    
    # ============================================
    # 2. FAMILLES DE VARIABLES
    # ============================================
    st.subheader("🧬 Étape 2 : Familles de variables")
    
    col_fam1, col_fam2, col_fam3 = st.columns(3)
    
    with col_fam1:
        st.markdown(f"**M1 — Transactionnelles** ({len(FEATURE_SETS['M1'])} variables)")
        with st.expander("Voir les variables", expanded=False):
            st.caption(", ".join(FEATURE_SETS["M1"]))
    
    with col_fam2:
        st.markdown(f"**M2 — + Comportementales** ({len(FEATURE_SETS['M2'])} variables)")
        with st.expander("Voir les variables", expanded=False):
            st.caption(", ".join(FEATURE_SETS["M2"]))
    
    with col_fam3:
        st.markdown(f"**M3 — + Temporelles** ({len(FEATURE_SETS['M3'])} variables)")
        with st.expander("Voir les variables", expanded=False):
            st.caption(", ".join(FEATURE_SETS["M3"]))
    
    # ============================================
    # 3. PARAMÈTRES D'ENTRAÎNEMENT
    # ============================================
    st.subheader("⚙️ Étape 3 : Paramètres d'entraînement")
    
    col_param1, col_param2, col_param3 = st.columns(3)
    
    with col_param1:
        contamination = st.slider(
            "Contamination (proportion attendue d'anomalies)",
            min_value=0.001,
            max_value=0.10,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Pourcentage estimé de transactions frauduleuses dans le dataset"
        )
    
    with col_param2:
        n_estimators = st.slider(
            "Nombre d'arbres dans Isolation Forest",
            min_value=50,
            max_value=300,
            value=100,
            step=10,
            help="Plus il y a d'arbres, plus le modèle est robuste mais plus lent à entraîner"
        )
    
    with col_param3:
        random_state = st.number_input(
            "random_state (graine aléatoire)",
            min_value=0,
            value=42,
            step=1,
            help="Fixer pour des résultats reproductibles"
        )
    
    # ============================================
    # 4. ENTRAÎNEMENT
    # ============================================
    st.subheader("🚀 Étape 4 : Entraînement des modèles")
    
    if st.button("🔄 Lancer l'entraînement des 3 modèles", type="primary", use_container_width=True):
        
        # Barre de progression
        progress_bar = st.progress(0)
        status_text = st.empty()
        logs = st.empty()
        
        # Stockage des résultats
        model_pipelines = {}
        n_components_info = {}
        scaler_m3 = None
        
        try:
            # Entraînement des 3 modèles
            for i, (name, cols) in enumerate(FEATURE_SETS.items()):
                status_text.info(f"🌲 Entraînement du modèle {name}...")
                logs.info(f"Variables utilisées : {len(cols)} colonnes")
                
                # Construction du pipeline
                pipeline = build_anomaly_pipeline(
                    cols,
                    contamination=contamination,
                    random_state=random_state
                )
                
                # Modifier le nombre d'arbres
                pipeline.named_steps['isolation_forest'].n_estimators = n_estimators
                
                # Entraînement
                with st.spinner(f"Entraînement de {name} en cours..."):
                    pipeline.fit(df)
                
                # Sauvegarde du scaler pour M3
                if name == "M3":
                    scaler_m3 = pipeline.named_steps['normalisation']
                    sauvegarder_modele(pipeline, name, scaler_m3)
                else:
                    sauvegarder_modele(pipeline, name)
                
                # Prédictions
                df[f"Score_{name}"] = pipeline.decision_function(df)
                df[f"Prediction_{name}"] = pipeline.predict(df)
                
                # Informations
                n_components_info[name] = pipeline.named_steps["acp"].n_components_
                
                # Statistiques
                n_anomalies = (df[f"Prediction_{name}"] == -1).sum()
                logs.info(f"✅ {name} terminé : {n_anomalies:,} anomalies détectées "
                         f"({n_anomalies/len(df)*100:.2f}%)")
                
                progress_bar.progress(int((i + 1) / len(FEATURE_SETS) * 100))
            
            # Succès
            status_text.success("✅ Entraînement terminé avec succès !")
            logs.success("Tous les modèles sont prêts à être utilisés.")
            
            # ============================================
            # 5. COMPARAISON DES MODÈLES
            # ============================================
            st.subheader("📊 Étape 5 : Comparaison des modèles")
            
            # Métriques de comparaison
            col_comp1, col_comp2, col_comp3 = st.columns(3)
            
            for idx, name in enumerate(FEATURE_SETS):
                with [col_comp1, col_comp2, col_comp3][idx]:
                    n_anomalies = (df[f"Prediction_{name}"] == -1).sum()
                    n_normales = (df[f"Prediction_{name}"] == 1).sum()
                    
                    st.markdown(f"**Modèle {name}**")
                    st.metric(
                        "🔴 Anomalies",
                        f"{n_anomalies:,}",
                        delta=f"{n_anomalies/len(df)*100:.2f}%"
                    )
                    st.metric(
                        "🟢 Normales",
                        f"{n_normales:,}",
                        delta=f"{n_normales/len(df)*100:.2f}%"
                    )
                    st.caption(f"PCA : {n_components_info[name]} composantes")
            
            # Tableau récapitulatif
            st.subheader("📈 Détail des scores")
            
            tab_comp1, tab_comp2, tab_comp3, tab_comp4 = st.tabs([
                "📋 Distribution", "📊 Scores", "🔍 Aperçu", "📉 Statistiques"
            ])
            
            with tab_comp1:
                dist_rows = []
                for name in FEATURE_SETS:
                    counts = df[f"Prediction_{name}"].value_counts()
                    dist_rows.append({
                        "Modèle": name,
                        "Normales": int(counts.get(1, 0)),
                        "Anomalies": int(counts.get(-1, 0)),
                        "% Anomalies": f"{counts.get(-1, 0) / len(df) * 100:.2f}%",
                        "PCA comp.": n_components_info[name]
                    })
                st.dataframe(pd.DataFrame(dist_rows), use_container_width=True)
            
            with tab_comp2:
                # Distribution des scores - CORRECTION ICI
                fig, ax = plt.subplots(figsize=(12, 6))
                for name in FEATURE_SETS:
                    sns.kdeplot(df[f"Score_{name}"], label=name, ax=ax)
                ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Seuil')
                ax.set_xlabel("Score d'anomalie")
                ax.set_ylabel("Densité")
                ax.set_title("Distribution des scores par modèle")
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
            
            with tab_comp3:
                preview_cols = ["amount", "type"] if "type" in df.columns else ["amount"]
                preview_cols += [f"Score_{n}" for n in FEATURE_SETS] + [f"Prediction_{n}" for n in FEATURE_SETS]
                st.dataframe(df[preview_cols].head(20), use_container_width=True)
            
            with tab_comp4:
                score_cols = [f"Score_{n}" for n in FEATURE_SETS]
                st.dataframe(df[score_cols].describe(), use_container_width=True)
            
            # ============================================
            # 6. TÉLÉCHARGEMENT
            # ============================================
            st.subheader("📥 Étape 6 : Télécharger le dataset enrichi")
            
            # Ajouter une colonne de sévérité pour faciliter l'analyse
            df["NbModelesAnomalie"] = (
                (df["Prediction_M1"] == -1).astype(int) +
                (df["Prediction_M2"] == -1).astype(int) +
                (df["Prediction_M3"] == -1).astype(int)
            )
            
            # Boutons de téléchargement
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Télécharger df_traite.csv",
                    data=csv,
                    file_name="df_traite.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption("💡 Utilisez ce fichier dans PY_03_Dashboard.py")
            
            with col_dl2:
                # Télécharger un échantillon pour test
                sample_size = min(1000, len(df))
                csv_sample = df.sample(n=sample_size, random_state=42).to_csv(index=False)
                st.download_button(
                    label=f"📥 Télécharger échantillon ({sample_size:,} lignes)",
                    data=csv_sample,
                    file_name="df_traite_echantillon.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption("🔬 Pour des tests rapides dans le dashboard")
            
            # ============================================
            # 7. RÉSUMÉ FINAL
            # ============================================
            st.success("""
            ✅ **Entraînement terminé avec succès !**
            
            **Résumé :**
            - 3 modèles entraînés (M1, M2, M3)
            - Modèles sauvegardés dans le dossier `models/`
            - Scaler M3 sauvegardé pour réutilisation
            - Dataset enrichi prêt pour le dashboard
            
            **Prochaine étape :** Lancez **PY_03_Dashboard.py** avec le fichier `df_traite.csv`
            """)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement : {str(e)}")
            st.exception(e)

else:
    st.info("👆 Chargez votre fichier `dataset_prepare.csv` pour commencer l'entraînement")
    st.markdown("""
    ### 📋 Format attendu du fichier :
    Le fichier doit contenir au minimum les colonnes :
    - Toutes les colonnes transactionnelles, comportementales et temporelles
    - Généré par PY_01_Preparation.py
    """)

st.divider()
st.caption("🤖 Entraînement des modèles - Projet PaySim Haïti")