"""
PY_01_Preparation.py
Prépare les données - Version alignée sur PaySim_pipelines_avance.ipynb

Ce script reproduit EXACTEMENT la séquence du pipeline scikit-learn du
notebook (TypeFilter -> AmountRangeFilter -> FeeCalculator ->
BalanceRecalculator -> BalanceSanityFilter -> TemporalFeatureCreator ->
DrainedAccountFlagger -> RatioVariationCalculator ->
BehavioralFeatureEngineer -> CategoricalEncoder), afin que le dataset
produit ici soit identique à celui utilisé plus loin pour l'entraînement
des modèles (IsolationForest / clustering / etc.).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="📊 Préparation des données",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Préparation des données PaySim")

st.markdown("""
Cette application reproduit fidèlement la phase de préparation du
notebook `PaySim_pipelines_avance.ipynb` :
1. **Charger** votre fichier CSV PaySim
2. **Filtrer** les DEBIT et les montants aberrants
3. **Calculer** les frais selon le barème métier
4. **Recalculer** les soldes (newbalanceOrig / newbalanceDest)
5. **Filtrer** les soldes incohérents
6. **Créer** les features temporelles et comportementales
7. **Sauvegarder** le dataset préparé
""")

# ============================================
# FONCTIONS — traduction 1:1 des transformers du notebook
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
    """Équivalent de TypeFilter."""
    return df[~df["type"].isin(exclude)].copy()


def filter_amount_range(df, low=10, high=100_000):
    """Équivalent de AmountRangeFilter."""
    return df[df["amount"].between(low, high)].copy()


def compute_fees(df):
    """Équivalent de FeeCalculator."""
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
    """Équivalent de BalanceRecalculator : écrase newbalanceOrig / newbalanceDest."""
    df = df.copy()
    df["newbalanceOrig"] = df["oldbalanceOrg"] - (df["amount"] + df["frais"])
    df["newbalanceDest"] = df["oldbalanceDest"] + df["amount"]
    return df


def filter_balance_sanity(df, cap=100_000):
    """Équivalent de BalanceSanityFilter."""
    df = df.copy()
    is_dest_client = df["nameDest"].str[0] == "C"
    is_orig_client = df["nameOrig"].str[0] == "C"

    mask = df["newbalanceOrig"] > 0
    mask &= ~(is_dest_client & (df["newbalanceDest"] > cap))
    mask &= ~(is_orig_client & (df["newbalanceOrig"] > cap))
    mask &= ~(is_orig_client & (df["oldbalanceOrg"] > cap))
    return df[mask]


def add_temporal_features(df, night_start=22, night_end=5):
    """Équivalent de TemporalFeatureCreator."""
    df = df.copy()
    df["heure"] = (df["step"] - 1) % 24
    df["jour"] = ((df["step"] - 1) // 24) + 1
    df["step_night"] = ((df["heure"] >= night_start) | (df["heure"] <= night_end)).astype(int)
    return df


def flag_drained_accounts(df):
    """Équivalent de DrainedAccountFlagger."""
    df = df.copy()
    df["is_drained"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] < 10)).astype(int)
    return df


def add_ratio_variation(df):
    """Équivalent de RatioVariationCalculator."""
    df = df.copy()
    df["ratio_amount_balance"] = np.where(df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"], 0)
    df["variationOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["variationDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
    return df


def add_behavioral_features(df):
    """Équivalent de BehavioralFeatureEngineer."""
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
    """Équivalent de CategoricalEncoder."""
    df = df.copy()
    df["type_encoded"] = df["type"].astype("category").cat.codes
    return df


def run_preparation_pipeline(df):
    """Enchaîne les étapes exactement dans l'ordre du preparation_pipeline du notebook."""
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
# 1. UPLOAD DU FICHIER
# ============================================

st.subheader("📂 Étape 1 : Chargement du fichier")

uploaded_file = st.file_uploader(
    "Choisissez votre fichier CSV PaySim",
    type=['csv'],
    help="Le fichier doit être au format PaySim standard"
)

if uploaded_file is not None:
    st.success(f"✅ Fichier chargé : {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")

    # Chargement robuste (gestion encodage, comme load_paysim du notebook)
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="latin-1")
    except Exception:
        st.error("❌ Impossible de lire le fichier CSV. Vérifiez le format.")
        st.stop()

    # Nettoyage minimal des noms de colonnes (BOM éventuel)
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')

    st.write("🔍 **Colonnes trouvées :**", df.columns.tolist())
    st.write("**Aperçu du fichier :**")
    st.dataframe(df.head(5), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Lignes", f"{len(df):,}")
    with col2:
        st.metric("📋 Colonnes", f"{len(df.columns)}")
    with col3:
        fraud_count = df['isFraud'].sum() if 'isFraud' in df.columns else 0
        st.metric("🚨 Transactions frauduleuses (retirées du pipeline)", f"{fraud_count:,}")

    if st.button("🚀 Lancer la préparation", type="primary"):

        required_cols = {"type", "amount", "step", "nameOrig", "nameDest",
                          "oldbalanceOrg", "oldbalanceDest"}
        missing = required_cols - set(df.columns)
        if missing:
            st.error(f"❌ Colonnes manquantes pour reproduire le pipeline du notebook : {sorted(missing)}")
            st.stop()

        with st.spinner("Préparation des données en cours (pipeline notebook)..."):

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Comme load_paysim : on retire isFraud / isFlaggedFraud avant le pipeline
            status_text.text("🧹 Retrait des colonnes cibles (isFraud, isFlaggedFraud)...")
            df = df.drop(columns=["isFraud", "isFlaggedFraud"], errors="ignore")
            progress_bar.progress(10)

            status_text.text("🧹 Filtrage type (DEBIT) et montant...")
            df = filter_type(df, exclude=("DEBIT",))
            df = filter_amount_range(df, low=10, high=100_000)
            progress_bar.progress(25)

            status_text.text("💰 Calcul des frais selon le barème métier...")
            df = compute_fees(df)
            progress_bar.progress(40)

            status_text.text("🏦 Recalcul des soldes (newbalanceOrig / newbalanceDest)...")
            df = recalculate_balances(df)
            df = filter_balance_sanity(df, cap=100_000)
            progress_bar.progress(55)

            status_text.text("⏰ Features temporelles...")
            df = add_temporal_features(df, night_start=22, night_end=5)
            df = flag_drained_accounts(df)
            df = add_ratio_variation(df)
            progress_bar.progress(70)

            status_text.text("🔄 Features comportementales par client/jour/heure...")
            df = add_behavioral_features(df)
            progress_bar.progress(90)

            status_text.text("🏷️ Encodage du type...")
            df = encode_type(df)
            progress_bar.progress(95)

            status_text.text("💾 Sauvegarde du dataset préparé...")
            os.makedirs("data", exist_ok=True)
            output_path = os.path.join("data", "dataset_prepare.csv")
            df.to_csv(output_path, index=False)
            progress_bar.progress(100)
            status_text.text("✅ Préparation terminée !")

        st.success("✅ Préparation terminée avec succès ! Le dataset correspond au pipeline du notebook.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Lignes", f"{len(df):,}")
        with col2:
            st.metric("📋 Colonnes", f"{len(df.columns)}")
        with col3:
            st.metric("💰 Frais total", f"{df['frais'].sum():,.0f} FCFA")
        with col4:
            st.metric("📈 Montant total", f"{df['amount'].sum():,.0f} FCFA")

        st.subheader("📊 Statistiques des données préparées")
        tab1, tab2, tab3 = st.tabs(["📈 Statistiques", "📋 Distribution par type", "🔍 Aperçu"])

        with tab1:
            st.dataframe(df.describe(), use_container_width=True)

        with tab2:
            type_dist = df["type"].value_counts().rename_axis("Type").reset_index(name="Count")
            type_dist["Percentage"] = (type_dist["Count"] / len(df) * 100).round(1).astype(str) + "%"
            st.dataframe(type_dist, use_container_width=True)

        with tab3:
            st.dataframe(df.head(10), use_container_width=True)

        st.subheader("📥 Télécharger les données préparées")
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger dataset_prepare.csv",
            data=csv,
            file_name="dataset_prepare.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.info("🚀 **Prochaine étape :** Exécutez **PY_02_Training.py** pour entraîner les modèles d'anomalie.")

else:
    st.info("👆 Chargez votre fichier CSV PaySim pour commencer")

st.divider()
st.caption("📊 Préparation des données - Projet PaySim Haïti (alignée sur le notebook)")
