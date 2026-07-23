"""
PY_04_Simulator.py
Simulateur de transactions + détection en temps réel — version autonome
(upload direct de dataset_prepare.csv, comme PY_01/PY_02/PY_03).

Logique :
1. On charge dataset_prepare.csv (sortie de PY_01) comme "base de référence".
2. On entraîne les 3 pipelines M1/M2/M3 dessus (même code que PY_02).
3. Pour chaque nouvelle transaction (saisie manuelle ou générée aléatoirement) :
   a. On la combine avec un échantillon de la base de référence (colonnes brutes)
   b. On refait tourner EXACTEMENT le même pipeline de préparation que PY_01
      dessus (pour recalculer frais, soldes, features comportementales...)
   c. On extrait la ligne recalculée de la nouvelle transaction
   d. On la score avec les 3 pipelines entraînés → Normal ou Anomalie

Entrée attendue : dataset_prepare.csv produit par PY_01_Preparation.py.
"""

import time

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="🧪 Simulateur temps réel", page_icon="🧪", layout="wide")
st.title("🧪 Simulateur & Détection en temps réel — PaySim")

st.markdown("""
Insérez une transaction (manuellement ou en lot simulé) et voyez
**immédiatement** ce que décident les 3 modèles entraînés (M1/M2/M3),
comme le ferait un système de détection en production.
""")

RAW_COLUMNS = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "nameDest", "oldbalanceDest"]
SEVERITY_MAP = {3: "🔴 Critique", 2: "🟠 Élevée", 1: "🟡 Modérée", 0: "🟢 Normale"}

# ============================================
# PIPELINE DE PRÉPARATION — identique à PY_01_Preparation.py
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
    amount, ttype = df["amount"], df["type"]
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
# FAMILLES DE VARIABLES ET PIPELINES DE MODÈLES — identiques à PY_02
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
        ("isolation_forest", IsolationForest(contamination=contamination, random_state=random_state)),
    ])


# ============================================
# GÉNÉRATION DE TRANSACTIONS SIMULÉES
# ============================================

_TYPE_WEIGHTS = {"CASH_OUT": 0.34, "PAYMENT": 0.33, "CASH_IN": 0.22, "TRANSFER": 0.09, "DEBIT": 0.02}


def _random_account_id(rng, prefix="C"):
    return f"{prefix}{rng.integers(10_000_000, 99_999_999)}"


def generate_synthetic_transactions(n=1, start_step=1, anomaly_bias=0.0, seed=None):
    """Génère n transactions brutes façon PaySim. anomaly_bias (0-1) contrôle
    la probabilité de générer un schéma suspect (montant proche du solde,
    heure de nuit) plutôt qu'un comportement tiré aléatoirement. Simplification
    à but de démonstration, ne reproduit pas le générateur statistique de PaySim."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ttype = rng.choice(list(_TYPE_WEIGHTS.keys()), p=list(_TYPE_WEIGHTS.values()))
        is_suspicious = rng.random() < anomaly_bias
        old_balance_org = float(rng.uniform(0, 50_000))

        if is_suspicious:
            amount = round(float(old_balance_org * rng.uniform(0.85, 1.15)), 2)
            step = int(start_step + i) if rng.random() < 0.5 else int(rng.integers(1, 30) * 24 - rng.integers(0, 6))
        else:
            amount = round(float(rng.uniform(10, 20_000)), 2)
            step = int(start_step + i)
        step = max(step, 1)

        old_balance_dest = float(rng.uniform(0, 50_000))
        name_orig = _random_account_id(rng, prefix="C")
        dest_prefix = "M" if (ttype == "PAYMENT" and rng.random() < 0.6) else "C"
        name_dest = _random_account_id(rng, prefix=dest_prefix)

        rows.append({
            "step": step, "type": ttype, "amount": amount,
            "nameOrig": name_orig, "oldbalanceOrg": round(old_balance_org, 2),
            "nameDest": name_dest, "oldbalanceDest": round(old_balance_dest, 2),
        })
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def score_new_transactions(new_tx_df, reference_df, model_pipelines, context_sample_size=5000, random_state=42):
    """Combine les nouvelles transactions avec un échantillon de la base de
    référence, refait tourner le pipeline de préparation, puis score les
    nouvelles transactions avec les pipelines de modèles entraînés.

    Limite assumée : les agrégats comportementaux (NbTransactionsJour, etc.)
    sont recalculés sur un ÉCHANTILLON de la référence, pas sa totalité,
    pour rester réactif en usage interactif."""
    new_tx_df = new_tx_df.copy().reset_index(drop=True)
    new_tx_df["_sim_id"] = [f"sim_{i}" for i in range(len(new_tx_df))]

    n_ref = len(reference_df)
    if n_ref > context_sample_size:
        reference_sample = reference_df[RAW_COLUMNS].sample(n=context_sample_size, random_state=random_state).copy()
    else:
        reference_sample = reference_df[RAW_COLUMNS].copy()
    reference_sample["_sim_id"] = np.nan

    combined = pd.concat([reference_sample, new_tx_df[RAW_COLUMNS + ["_sim_id"]]], ignore_index=True)
    prepared = run_preparation_pipeline(combined)
    sim_rows = prepared[prepared["_sim_id"].notna()].copy()
    kept_ids = set(sim_rows["_sim_id"])

    results = []
    for _, original in new_tx_df.iterrows():
        sim_id = original["_sim_id"]
        if sim_id not in kept_ids:
            results.append({"_sim_id": sim_id, "status": "rejected_by_pipeline_filters",
                             "input": original[RAW_COLUMNS].to_dict()})
            continue
        row = sim_rows[sim_rows["_sim_id"] == sim_id].iloc[[0]]
        entry = {"_sim_id": sim_id, "status": "scored",
                 "input": original[RAW_COLUMNS].to_dict(), "features": row.iloc[0].to_dict()}
        for name, pipeline in model_pipelines.items():
            entry[f"Score_{name}"] = float(pipeline.decision_function(row)[0])
            entry[f"Prediction_{name}"] = int(pipeline.predict(row)[0])
        results.append(entry)
    return results


# ============================================
# ÉTAPE 1 : CHARGEMENT DE LA BASE DE RÉFÉRENCE
# ============================================

st.subheader("📂 Étape 1 : Charger la base de référence")

uploaded_file = st.file_uploader(
    "Choisissez votre fichier dataset_prepare.csv (issu de PY_01_Preparation.py)",
    type=["csv"],
    help="Sert de contexte pour recalculer les features comportementales des nouvelles transactions"
)

if uploaded_file is None:
    st.info("👆 Chargez votre dataset_prepare.csv pour commencer")
    st.stop()

try:
    reference_df = pd.read_csv(uploaded_file, encoding="utf-8")
except UnicodeDecodeError:
    uploaded_file.seek(0)
    reference_df = pd.read_csv(uploaded_file, encoding="latin-1")
except Exception:
    st.error("❌ Impossible de lire le fichier CSV. Vérifiez le format.")
    st.stop()

reference_df.columns = reference_df.columns.str.strip().str.replace('\ufeff', '')
st.success(f"✅ Fichier chargé : {uploaded_file.name} — {len(reference_df):,} lignes")

missing = set(FEATURE_SETS["M3"]) - set(reference_df.columns)
if missing:
    st.error(f"❌ Colonnes manquantes : {sorted(missing)}. Vérifiez que ce fichier vient bien de PY_01_Preparation.py.")
    st.stop()

# ============================================
# ÉTAPE 2 : ENTRAÎNEMENT DES MODÈLES POUR LA SIMULATION
# ============================================

st.subheader("🤖 Étape 2 : Entraîner les modèles pour la simulation")

t1, t2 = st.columns(2)
with t1:
    contamination = st.slider("Contamination", 0.001, 0.10, 0.01, 0.001, format="%.3f")
with t2:
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)

if st.button("🚀 Entraîner M1 / M2 / M3", type="primary"):
    with st.spinner("Entraînement des 3 modèles sur la base de référence..."):
        model_pipelines = {
            name: build_anomaly_pipeline(cols, contamination=contamination, random_state=int(random_state))
            for name, cols in FEATURE_SETS.items()
        }
        for name, pipeline in model_pipelines.items():
            pipeline.fit(reference_df)
        st.session_state["model_pipelines"] = model_pipelines
        st.session_state["reference_df"] = reference_df
    st.success("✅ Modèles entraînés et prêts pour la simulation.")

if "model_pipelines" not in st.session_state:
    st.warning("⚠️ Entraînez d'abord les modèles ci-dessus pour activer la simulation.")
    st.stop()

model_pipelines = st.session_state["model_pipelines"]
reference_df = st.session_state["reference_df"]

with st.expander("⚙️ Paramètres avancés"):
    context_sample_size = st.slider(
        "Taille de l'échantillon de contexte (agrégats comportementaux)",
        500, 20000, 5000, 500,
        help="Un échantillon de la base de référence est combiné à chaque nouvelle transaction "
             "pour recalculer ses agrégats. Plus petit = plus rapide, moins représentatif."
    )

st.divider()

# ============================================
# JOURNAL DES TESTS (SESSION)
# ============================================

if "alert_log" not in st.session_state:
    st.session_state.alert_log = []


def log_result(result, source):
    if result["status"] == "scored":
        nb_alert = sum(1 for n in FEATURE_SETS if result.get(f"Prediction_{n}") == -1)
        row = {
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "source": source,
            "type": result["input"]["type"],
            "amount": result["input"]["amount"],
            "Score_M1": round(result.get("Score_M1", np.nan), 4),
            "Score_M2": round(result.get("Score_M2", np.nan), 4),
            "Score_M3": round(result.get("Score_M3", np.nan), 4),
            "NbModelesAnomalie": nb_alert,
            "Sévérité": SEVERITY_MAP[nb_alert],
        }
    else:
        row = {
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "source": source,
            "type": result["input"]["type"],
            "amount": result["input"]["amount"],
            "Score_M1": None, "Score_M2": None, "Score_M3": None,
            "NbModelesAnomalie": None,
            "Sévérité": "⚪ Rejetée par le pipeline",
        }
    st.session_state.alert_log.insert(0, row)
    st.session_state.alert_log = st.session_state.alert_log[:300]


def render_result_card(result):
    if result["status"] == "rejected_by_pipeline_filters":
        st.warning(
            "⚪ **Transaction rejetée par les filtres du pipeline** (type DEBIT, montant hors "
            "[10, 100000], ou solde incohérent après recalcul) — comportement attendu, elle n'a "
            "donc pas pu être scorée par les modèles."
        )
        st.json(result["input"])
        return

    nb_alert = sum(1 for n in FEATURE_SETS if result.get(f"Prediction_{n}") == -1)
    severity = SEVERITY_MAP[nb_alert]
    if nb_alert == 3:
        st.error(f"{severity} — les 3 modèles jugent cette transaction anormale.")
    elif nb_alert == 2:
        st.warning(f"{severity} — 2 modèles sur 3 jugent cette transaction anormale.")
    elif nb_alert == 1:
        st.info(f"{severity} — 1 modèle sur 3 juge cette transaction anormale.")
    else:
        st.success(f"{severity} — aucun modèle ne détecte d'anomalie.")

    cols = st.columns(3)
    for col, name in zip(cols, FEATURE_SETS):
        with col:
            pred = result.get(f"Prediction_{name}")
            score = result.get(f"Score_{name}")
            label = "🔴 Anomalie" if pred == -1 else "🟢 Normale"
            st.metric(f"Modèle {name}", label, delta=f"score {score:.4f}")

    with st.expander("Voir les features calculées pour cette transaction"):
        feat_cols = [c for c in FEATURE_SETS["M3"] if c in result["features"]]
        st.dataframe(pd.DataFrame([{c: result["features"][c] for c in feat_cols}]), use_container_width=True)


# ============================================
# SIMULATION — SAISIE MANUELLE / LOT / FLUX TEMPS RÉEL
# ============================================

tab_manual, tab_batch, tab_live = st.tabs(
    ["✍️ Saisie manuelle", "🎲 Génération en lot", "🔴 Flux temps réel"]
)

with tab_manual:
    st.subheader("✍️ Tester une transaction précise")
    default_step = int(reference_df["step"].max()) + 1 if "step" in reference_df.columns else 1

    f1, f2, f3 = st.columns(3)
    with f1:
        m_type = st.selectbox("Type", ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"])
        m_amount = st.number_input("Montant", min_value=0.0, value=5000.0, step=100.0)
    with f2:
        m_name_orig = st.text_input("Compte émetteur (nameOrig)", value="C12345678")
        m_old_balance_org = st.number_input("Solde émetteur avant", min_value=0.0, value=10000.0, step=100.0)
    with f3:
        m_name_dest = st.text_input("Compte destinataire (nameDest)", value="C87654321")
        m_old_balance_dest = st.number_input("Solde destinataire avant", min_value=0.0, value=2000.0, step=100.0)
    m_step = st.number_input("Step (1h = 1 step)", min_value=1, value=default_step)

    if st.button("🧪 Tester cette transaction", type="primary"):
        manual_tx = pd.DataFrame([{
            "step": m_step, "type": m_type, "amount": m_amount,
            "nameOrig": m_name_orig, "oldbalanceOrg": m_old_balance_org,
            "nameDest": m_name_dest, "oldbalanceDest": m_old_balance_dest,
        }])[RAW_COLUMNS]

        with st.spinner("Scoring..."):
            results = score_new_transactions(manual_tx, reference_df, model_pipelines,
                                              context_sample_size=context_sample_size)
            log_result(results[0], source="manuel")

        st.subheader("📋 Résultat")
        render_result_card(results[0])

with tab_batch:
    st.subheader("🎲 Générer et tester un lot de transactions")
    b1, b2 = st.columns(2)
    with b1:
        n_batch = st.number_input("Nombre de transactions", min_value=1, max_value=500, value=20)
    with b2:
        bias_batch = st.slider("Biais vers des schémas suspects", 0.0, 1.0, 0.3, 0.05)

    if st.button("🎲 Générer et tester le lot", type="primary"):
        with st.spinner(f"Génération et scoring de {n_batch} transactions..."):
            default_step = int(reference_df["step"].max()) + 1 if "step" in reference_df.columns else 1
            batch_tx = generate_synthetic_transactions(n=int(n_batch), start_step=default_step, anomaly_bias=bias_batch)
            results = score_new_transactions(batch_tx, reference_df, model_pipelines,
                                              context_sample_size=context_sample_size)
            for r in results:
                log_result(r, source="lot")

        summary_rows = []
        for r in results:
            if r["status"] == "scored":
                nb_alert = sum(1 for n in FEATURE_SETS if r.get(f"Prediction_{n}") == -1)
                summary_rows.append({"type": r["input"]["type"], "amount": r["input"]["amount"],
                                      "Score_M3": round(r.get("Score_M3", np.nan), 4),
                                      "NbModelesAnomalie": nb_alert, "Sévérité": SEVERITY_MAP[nb_alert]})
            else:
                summary_rows.append({"type": r["input"]["type"], "amount": r["input"]["amount"],
                                      "Score_M3": None, "NbModelesAnomalie": None,
                                      "Sévérité": "⚪ Rejetée par le pipeline"})
        summary_df = pd.DataFrame(summary_rows)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Testées", len(summary_df))
        with s2:
            st.metric("🔴 Critiques", int((summary_df["NbModelesAnomalie"] == 3).sum()))
        with s3:
            st.metric("🟠 Élevées", int((summary_df["NbModelesAnomalie"] == 2).sum()))
        with s4:
            st.metric("⚪ Rejetées", int(summary_df["Sévérité"].eq("⚪ Rejetée par le pipeline").sum()))

        st.bar_chart(summary_df["Sévérité"].value_counts())
        st.dataframe(summary_df.sort_values("Score_M3", na_position="last"), use_container_width=True)

with tab_live:
    st.subheader("🔴 Flux de transactions simulées en temps réel")
    st.caption(
        "Génère des transactions une à une et les affiche au fur et à mesure qu'elles sont "
        "scorées, avec alerte immédiate en cas d'anomalie."
    )

    l1, l2, l3 = st.columns(3)
    with l1:
        n_stream = st.number_input("Nombre de transactions à simuler", 1, 200, 15, key="live_n")
    with l2:
        delay = st.slider("Délai entre transactions (s)", 0.0, 3.0, 0.4, 0.1, key="live_delay")
    with l3:
        bias_live = st.slider("Biais vers des schémas suspects", 0.0, 1.0, 0.25, 0.05, key="live_bias")

    start = st.button("▶️ Démarrer le flux", type="primary")
    alert_banner = st.empty()
    live_table_placeholder = st.empty()

    if start:
        default_step = int(reference_df["step"].max()) + 1 if "step" in reference_df.columns else 1
        new_alerts = 0
        for i in range(int(n_stream)):
            sim_tx = generate_synthetic_transactions(n=1, start_step=default_step + i, anomaly_bias=bias_live)
            results = score_new_transactions(sim_tx, reference_df, model_pipelines,
                                              context_sample_size=context_sample_size)
            result = results[0]
            log_result(result, source="flux temps réel")

            if result["status"] == "scored":
                nb_alert = sum(1 for n in FEATURE_SETS if result.get(f"Prediction_{n}") == -1)
                if nb_alert >= 2:
                    new_alerts += 1

            live_table_placeholder.dataframe(pd.DataFrame(st.session_state.alert_log[:50]), use_container_width=True)
            if new_alerts > 0:
                alert_banner.error(f"🚨 {new_alerts} alerte(s) élevée(s)/critique(s) détectée(s) dans ce flux !")
            time.sleep(delay)

        st.success(f"✅ Flux terminé — {n_stream} transactions simulées.")
    elif st.session_state.alert_log:
        live_table_placeholder.dataframe(pd.DataFrame(st.session_state.alert_log[:50]), use_container_width=True)

st.divider()

# ============================================
# HISTORIQUE COMPLET DE LA SESSION
# ============================================

st.subheader("📜 Historique des transactions testées (session en cours)")
if st.session_state.alert_log:
    log_df = pd.DataFrame(st.session_state.alert_log)
    st.dataframe(log_df, use_container_width=True)
    csv = log_df.to_csv(index=False)
    st.download_button("📥 Télécharger l'historique", data=csv, file_name="simulation_log.csv", mime="text/csv")
else:
    st.info("Aucune transaction testée pour l'instant.")

st.caption("🧪 Simulateur temps réel - Projet PaySim Haïti (aligné sur le notebook)")