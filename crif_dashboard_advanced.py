#!/usr/bin/env python3
"""
Dashboard Streamlit avanzata per dati "CRIF rating tutti" (dicembre 2025)
- Panoramica descrittiva
- Scostamenti agosto vs dicembre
- Clustering su dicembre
- Modello predittivo per rating futuro (proxy aprile)
"""

import os
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DEFAULT_FILE = "Esito CRIF Rating 2025-12.xlsx"
DEFAULT_SHEET = "CRIF rating tutti"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = BASE_DIR / DEFAULT_FILE

st.set_page_config(page_title="CRIF Rating - Analisi Avanzata", layout="wide")
st.title("CRIF Rating - Dashboard Avanzata TFS")


@st.cache_data
def load_data(path: str, sheet: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet)


def ensure_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [col for col in required if col in df.columns]


def rating_color_map() -> dict:
    # Colori richiesti: 1-3 verdi, 4-5 gialli, 6-7 rossi, 10 nero, altri grigi
    return {
        "1": "#1F9D55",
        "2": "#27AE60",
        "3": "#2ECC71",
        "4": "#F1C40F",
        "5": "#F4D03F",
        "6": "#E74C3C",
        "7": "#C0392B",
        "10": "#000000",
        "Non valutabile": "#95A5A6",
        "Cessata": "#7F8C8D",
    }


def highlight_negative(val) -> str:
    try:
        return "color: red; font-weight: bold;" if float(val) < 0 else ""
    except Exception:
        return ""


def rating_band_color(rating: str) -> str:
    """Ritorna un colore di sfondo tenue in base alla classe di rating (per righe/colonne)."""
    rating = str(rating)
    if rating in {"1", "2", "3"}:
        return "#e9f6ec"  # verde chiaro
    if rating in {"4", "5"}:
        return "#fff7d6"  # giallo chiaro
    if rating in {"6", "7"}:
        return "#ffe9e9"  # rosso chiaro
    if rating in {"10"}:
        return "#f0f0f0"  # nero/grigio chiaro
    if rating == "Cessata":
        return "#f5f5f5"
    if rating == "Non valutabile":
        return "#f5f5f5"
    if rating == "New entry":
        return "#e8f2ff"  # azzurro tenue
    return ""


def rating_band_style_rows(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for idx in df.index:
        color = rating_band_color(idx)
        if color:
            styles.loc[idx, :] = f"background-color: {color};"
    return styles


def rating_band_style_cols(df: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    for col in df.columns:
        color = rating_band_color(col)
        if color:
            styles.loc[:, col] = f"background-color: {color};"
    return styles


# ===== Configurazione e caricamento dati =====
st.sidebar.header("Configurazione dati")
file_exists = os.path.exists(DEFAULT_FILE)
df = None
user_file = st.sidebar.file_uploader("Carica un file Excel", type=["xlsx", "xls"])

if user_file:
    df = pd.read_excel(user_file, sheet_name=DEFAULT_SHEET)
    st.sidebar.success(f"✅ File caricato: {user_file.name}")
elif DEFAULT_PATH.exists():
    try:
        df = load_data(DEFAULT_PATH, DEFAULT_SHEET)
        st.sidebar.success(f"✅ File di default caricato: {DEFAULT_PATH.name}")
    except Exception as exc:  # pragma: no cover
        st.sidebar.error(f"Errore nel caricamento: {exc}")
else:
    st.sidebar.error(f"File di default non trovato: {DEFAULT_PATH}")
    st.stop()

if df is None:
    st.error("Nessun dato disponibile")
    st.stop()

st.sidebar.info(f"Righe: {len(df)} | Colonne: {len(df.columns)}")

# ===== Sezioni principali =====
pan_tab, delta_tab, cluster_tab, model_tab, data_tab = st.tabs(
    ["Panoramica", "Scostamenti Ago→Dic", "Clustering Dic", "Predizione Aprile", "Dati"]
)


with pan_tab:
    # ----- Panoramica -----
    st.subheader("Panoramica dati dicembre")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Contratti", len(df))
    with col2:
        esposizione_tot = df["Esposizione"].sum(skipna=True)
        st.metric("Esposizione totale", f"€ {esposizione_tot:,.0f}")
    with col3:
        overdue_tot = df["Overdue"].sum(skipna=True)
        st.metric("Overdue totale", f"€ {overdue_tot:,.0f}")
    with col4:
        missing = int(df.isnull().sum().sum())
        st.metric("Valori mancanti", missing)

    # Distribuzione rating dicembre
    if "CRIF Rating" in df.columns:
        rating_order = ["1", "2", "3", "4", "5", "6", "7", "10", "Non valutabile", "Cessata", "New entry"]
        rating_series = df["CRIF Rating"].astype(str)
        rating_cat = pd.Categorical(rating_series, categories=rating_order, ordered=True)
        rating_counts = (
            pd.DataFrame({"Rating": rating_cat})
            .groupby("Rating", observed=False)
            .size()
            .reset_index(name="Count")
            .dropna(subset=["Rating"])
        )
        rating_present = [r for r in rating_order if r in set(rating_counts["Rating"].tolist())]
        fig_rating = px.bar(
            rating_counts,
            x="Rating",
            y="Count",
            title="Distribuzione CRIF Rating dicembre",
            color="Rating",
            text_auto=True,
            color_discrete_map=rating_color_map(),
            category_orders={"Rating": rating_present},
        )
        fig_rating.update_xaxes(
            categoryorder="array",
            categoryarray=rating_present,
            tickmode="array",
            tickvals=rating_present,
            ticktext=rating_present,
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    # Esposizione per brand
    if "Brand" in df.columns:
        expo_brand = df.groupby("Brand")["Esposizione"].sum().reset_index()
        fig_brand = px.bar(
            expo_brand,
            x="Brand",
            y="Esposizione",
            title="Esposizione per Brand",
            text_auto=".2s",
        )
        st.plotly_chart(fig_brand, use_container_width=True)

    # Scatter Esposizione vs Overdue con filtri
    if {"Esposizione", "Overdue"}.issubset(df.columns):
        color_map = rating_color_map()
        color_series = df.get("CRIF Rating", pd.Series(dtype=str)).astype(str)
        rating_order = ["1", "2", "3", "4", "5", "6", "7", "10", "Non valutabile", "Cessata", "New entry"]

        # Filtri
        colf1, colf2 = st.columns(2)
        rating_opts = color_series.dropna().unique().tolist()
        rating_present = [r for r in rating_order if r in rating_opts]
        if not rating_present:
            rating_present = sorted(rating_opts)
        brand_opts = sorted(df["Brand"].dropna().unique().tolist()) if "Brand" in df.columns else []

        with colf1:
            rating_sel = st.multiselect(
                "Filtra rating",
                options=rating_opts,
                default=rating_opts,
                key="scatter_rating_filter",
            )
        with colf2:
            brand_sel = st.multiselect(
                "Filtra brand",
                options=brand_opts,
                default=brand_opts,
                key="scatter_brand_filter",
            )

        # Range filter per Esposizione e Overdue
        min_exp, max_exp = float(df["Esposizione"].min()), float(df["Esposizione"].max())
        min_ovd, max_ovd = float(df["Overdue"].min()), float(df["Overdue"].max())
        colr1, colr2 = st.columns(2)
        with colr1:
            exp_range = st.slider(
                "Range Esposizione (€)",
                min_value=min_exp,
                max_value=max_exp,
                value=(min_exp, max_exp),
                step=(max_exp - min_exp) / 100 if max_exp > min_exp else 1.0,
                format="%.0f",
                key="scatter_exp_range",
            )
            st.caption(f"Esposizione: {exp_range[0]:,.0f} € → {exp_range[1]:,.0f} €")
        with colr2:
            ovd_range = st.slider(
                "Range Overdue (€)",
                min_value=min_ovd,
                max_value=max_ovd,
                value=(min_ovd, max_ovd),
                step=(max_ovd - min_ovd) / 100 if max_ovd > min_ovd else 1.0,
                format="%.0f",
                key="scatter_ovd_range",
            )
            st.caption(f"Overdue: {ovd_range[0]:,.0f} € → {ovd_range[1]:,.0f} €")

        # Scala log opzionale
        use_log_x = st.checkbox("Scala log su Esposizione", value=False, key="scatter_log_x")
        use_log_y = st.checkbox("Scala log su Overdue", value=False, key="scatter_log_y")

        df_scatter = df.copy()
        if rating_sel:
            df_scatter = df_scatter[df_scatter["CRIF Rating"].astype(str).isin(rating_sel)]
        if brand_sel and "Brand" in df_scatter.columns:
            df_scatter = df_scatter[df_scatter["Brand"].isin(brand_sel)]
        df_scatter = df_scatter[
            (df_scatter["Esposizione"].between(exp_range[0], exp_range[1], inclusive="both"))
            & (df_scatter["Overdue"].between(ovd_range[0], ovd_range[1], inclusive="both"))
        ]

        fig_scatter = px.scatter(
            df_scatter,
            x="Esposizione",
            y="Overdue",
            color=df_scatter.get("CRIF Rating", pd.Series(dtype=str)).astype(str),
            hover_data=[col for col in ["CUSTOMER_CODE", "Brand", "TIPOANAG"] if col in df_scatter.columns],
            title="Esposizione vs Overdue (colorato per rating)",
            labels={"color": "CRIF Rating"},
            color_discrete_map=color_map,
            category_orders={"CRIF Rating": rating_present},
        )
        if use_log_x:
            fig_scatter.update_xaxes(type="log")
        if use_log_y:
            fig_scatter.update_yaxes(type="log")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.subheader("Focus rischio e concentrazione")

    # Istogramma variazione rating
    if "Variazione CRIF" in df.columns:
        fig_var = px.histogram(
            df,
            x="Variazione CRIF",
            nbins=30,
            title="Distribuzione variazione rating (dic - ago)",
        )
        st.plotly_chart(fig_var, use_container_width=True, key="variazione_crif_hist")

    # Overdue rate per brand
    if {"Brand", "Esposizione", "Overdue"}.issubset(df.columns):
        brand_risk = df.groupby("Brand")[["Esposizione", "Overdue"]].sum().reset_index()
        brand_risk["Overdue_rate"] = brand_risk["Overdue"] / brand_risk["Esposizione"].replace(0, np.nan)
        fig_or = px.bar(
            brand_risk,
            x="Brand",
            y="Overdue_rate",
            title="Overdue rate per Brand",
            text_auto=".1%",
        )
        fig_or.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_or, use_container_width=True, key="overdue_rate_brand")

    # Heatmap rating vs cluster esposizione (se disponibili)
    if {"CRIF Rating", "Cluster Expo"}.issubset(df.columns):
        pivot = pd.pivot_table(
            df,
            index="CRIF Rating",
            columns="Cluster Expo",
            values="Esposizione" if "Esposizione" in df.columns else "CUSTOMER_CODE",
            aggfunc="sum",
            fill_value=0,
        )
        fig_heat = px.imshow(
            pivot,
            color_continuous_scale="Blues",
            title="Heatmap Esposizione per Rating e Cluster Expo",
            aspect="auto",
        )
        st.plotly_chart(fig_heat, use_container_width=True, key="heatmap_rating_cluster")

        # Heatmap aggiuntiva: conteggio clienti per rating e cluster
        count_values = "CUSTOMER_CODE" if "CUSTOMER_CODE" in df.columns else "CRIF Rating"
        pivot_count = pd.pivot_table(
            df,
            index="CRIF Rating",
            columns="Cluster Expo",
            values=count_values,
            aggfunc="count",
            fill_value=0,
        )
        fig_heat_count = px.imshow(
            pivot_count,
            color_continuous_scale="Blues",
            title="Heatmap conteggio clienti per Rating e Cluster Expo",
            aspect="auto",
        )
        st.plotly_chart(fig_heat_count, use_container_width=True, key="heatmap_rating_cluster_count")

    # Indicatori di concentrazione (HHI e share top 10/20)
    if "Esposizione" in df.columns:
        expo = df["Esposizione"].fillna(0).clip(lower=0)
        total = expo.sum()
        if total > 0:
            shares = (expo / total).sort_values(ascending=False)
            hhi = float((shares ** 2).sum())
            top10 = float(shares.head(10).sum())
            top20 = float(shares.head(20).sum())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("HHI esposizione", f"{hhi:.3f}")
            with col2:
                st.metric("Share top 10", f"{top10:.1%}")
            with col3:
                st.metric("Share top 20", f"{top20:.1%}")

    # Top 10 esposizione e peggiori per overdue
    if "Esposizione" in df.columns:
        top10_expo = df.nlargest(10, "Esposizione")[
            [c for c in ["CUSTOMER_CODE", "Brand", "TIPOANAG", "Esposizione", "Overdue", "CRIF Rating"] if c in df.columns]
        ]
        top10_expo_display = top10_expo.copy()
        for col in ["Esposizione", "Overdue"]:
            if col in top10_expo_display.columns:
                top10_expo_display[col] = top10_expo_display[col].map(lambda v: f"{v:,.0f}")
        st.write("Top 10 per Esposizione")
        st.dataframe(top10_expo_display, use_container_width=True)

    if "Overdue" in df.columns:
        worst_overdue = df.nlargest(10, "Overdue")[
            [c for c in ["CUSTOMER_CODE", "Brand", "TIPOANAG", "Overdue", "Esposizione", "CRIF Rating"] if c in df.columns]
        ]
        worst_overdue_display = worst_overdue.copy()
        for col in ["Esposizione", "Overdue"]:
            if col in worst_overdue_display.columns:
                worst_overdue_display[col] = worst_overdue_display[col].map(lambda v: f"{v:,.0f}")
        st.write("Peggiori 10 per Overdue")
        st.dataframe(worst_overdue_display, use_container_width=True)


with delta_tab:
    # ----- Scostamenti -----
    st.subheader("Scostamenti agosto → dicembre")
    req_cols = ensure_columns(df, ["CRIF Rating", "CRIF Rating 2025-08"])
    if len(req_cols) == 2:
        df_delta = df[req_cols + ["Esposizione", "Overdue", "Variazione CRIF"]].copy()
        df_delta["Cambio rating"] = df_delta["CRIF Rating"] != df_delta["CRIF Rating 2025-08"]

        cols_top = st.columns(6)
        changed = int(df_delta["Cambio rating"].sum())
        perc_change = 100 * changed / max(len(df_delta), 1)

        matrix = pd.crosstab(df_delta["CRIF Rating 2025-08"], df_delta["CRIF Rating"])
        # Rimuove righe/colonne totalmente vuote per una vista più pulita
        matrix = matrix.loc[(matrix.sum(axis=1) > 0), (matrix.sum(axis=0) > 0)]
        st.write("Matrice transizioni agosto → dicembre (conteggi)")

        # KPI invariati/migliorati/peggiorati
        rating_order = ["1", "2", "3", "4", "5", "6", "7", "10", "Non valutabile", "Cessata", "New entry"]
        rank = {r: i for i, r in enumerate(rating_order)}

        def rating_rank(val):
            return rank.get(str(val), len(rating_order))

        old_rank = df_delta["CRIF Rating 2025-08"].map(rating_rank)
        new_rank = df_delta["CRIF Rating"].map(rating_rank)
        invariati = int((old_rank == new_rank).sum())
        migliorati = int((new_rank < old_rank).sum())
        peggiorati = int((new_rank > old_rank).sum())

        with cols_top[0]:
            st.metric("Totale record", len(df_delta))
        with cols_top[1]:
            st.metric("Rating cambiati", changed)
        with cols_top[2]:
            st.metric("% cambiati", f"{perc_change:.1f}%")
        with cols_top[3]:
            st.metric("Invariati", invariati)
        with cols_top[4]:
            st.metric("Migliorati", migliorati)
        with cols_top[5]:
            st.metric("Peggiorati", peggiorati)

        diag_style = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
        for idx in matrix.index:
            if idx in matrix.columns:
                diag_style.loc[idx, idx] = "background-color: #d1f7c4; font-weight: bold;"

        styled_matrix = (
            matrix.style.apply(rating_band_style_cols, axis=None)
            .apply(lambda _: diag_style, axis=None)
        )
        st.dataframe(styled_matrix, use_container_width=True, height=720)

        # Matrice esposizione per transizioni (somma Esposizione)
        if "Esposizione" in df_delta.columns:
            matrix_exp = pd.pivot_table(
                df_delta,
                index="CRIF Rating 2025-08",
                columns="CRIF Rating",
                values="Esposizione",
                aggfunc="sum",
                fill_value=0,
            )
            # Mantieni anche valori negativi: rimuovi solo righe/colonne davvero tutte a zero
            matrix_exp = matrix_exp.loc[(matrix_exp != 0).any(axis=1), (matrix_exp != 0).any(axis=0)]

            diag_style_exp = pd.DataFrame("", index=matrix_exp.index, columns=matrix_exp.columns)
            for idx in matrix_exp.index:
                if idx in matrix_exp.columns:
                    diag_style_exp.loc[idx, idx] = "background-color: #d1f7c4; font-weight: bold;"

            styled_matrix_exp = (
                matrix_exp.style.format("{:,.2f}")
                .apply(rating_band_style_cols, axis=None)
                .apply(lambda _: diag_style_exp, axis=None)
                .applymap(highlight_negative)
            )
            st.write("Matrice transizioni agosto → dicembre (esposizione)")
            st.dataframe(styled_matrix_exp, use_container_width=True, height=720)

        if "Variazione CRIF" in df_delta.columns:
            fig_var = px.histogram(
                df_delta,
                x="Variazione CRIF",
                title="Distribuzione variazione rating (dic - ago)",
                nbins=30,
            )
            st.plotly_chart(fig_var, use_container_width=True)

        if "Esposizione" in df_delta.columns:
            expo_by_change = (
                df_delta.dropna(subset=["Esposizione"])
                .groupby("Cambio rating")["Esposizione"]
                .sum()
                .reset_index()
                .rename(columns={"Esposizione": "Esposizione_tot"})
            )
            expo_by_change["Cambio rating"] = expo_by_change["Cambio rating"].map({True: "Cambiati", False: "Invariati"})

            fig_delta_exp = px.histogram(
                df_delta.dropna(subset=["Esposizione"]),
                x="Esposizione",
                color="Cambio rating",
                title="Esposizione per stato cambio rating",
            )
            fig_delta_exp.update_layout(
                height=520,
                legend=dict(font=dict(size=12)),
            )
            # Numeri a supporto dell'istogramma (record ed esposizione) posizionati sopra il grafico
            # Metriche essenziali a supporto dell'istogramma
            m1, m2 = st.columns(2)
            chg_exp = expo_by_change.loc[expo_by_change["Cambio rating"] == "Cambiati", "Esposizione_tot"].sum()
            chg_count = int((df_delta["Cambio rating"] == True).sum())
            avg_ticket_chg = chg_exp / max(chg_count, 1)
            with m1:
                st.metric("Somma esposizione posizioni cambiate (€)", f"{chg_exp:,.0f} €")
            with m2:
                st.metric("Esposizione media per posizione cambiata (€)", f"{avg_ticket_chg:,.0f} €")
            st.caption(
                "Valori calcolati sul dataset filtrato: totale esposizione e ticket medio delle posizioni che hanno cambiato rating."
            )

            st.plotly_chart(fig_delta_exp, use_container_width=True)

            # Esposizione cambiati (verde/rosso) e invariati (grigio) affiancati
            exp_detail = df_delta.dropna(subset=["Esposizione"]).copy()
            exp_detail["old_rank"] = exp_detail["CRIF Rating 2025-08"].map(rating_rank)
            exp_detail["new_rank"] = exp_detail["CRIF Rating"].map(rating_rank)

            # Cambiati: split migliorati/peggiorati
            changed_detail = exp_detail[exp_detail["Cambio rating"]].copy()
            changed_detail["direzione"] = np.where(
                changed_detail["new_rank"] < changed_detail["old_rank"], "Migliorati", "Peggiorati"
            )
            exp_changed = (
                changed_detail.groupby("direzione")["Esposizione"]
                .sum()
                .reset_index()
                .rename(columns={"Esposizione": "Esposizione_tot"})
            )
            exp_changed["Stato"] = "Cambiati"

            # Invariati: singola barra grigia
            invariati_exp = (
                exp_detail.loc[~exp_detail["Cambio rating"], "Esposizione"].sum()
            )
            exp_invar = pd.DataFrame(
                {"direzione": ["Invariati"], "Esposizione_tot": [invariati_exp], "Stato": ["Invariati"]}
            )

            exp_bars = pd.concat([exp_changed, exp_invar], ignore_index=True)

            fig_delta_stack = px.bar(
                exp_bars,
                x="Stato",
                y="Esposizione_tot",
                color="direzione",
                color_discrete_map={"Migliorati": "#2ecc40", "Peggiorati": "#ff4136", "Invariati": "#999999"},
                title="Esposizione: invariati (grigio) e cambiati (verde/rosso)",
                text=exp_bars["Esposizione_tot"].map(lambda v: f"{v:,.0f} €"),
            )
            fig_delta_stack.update_layout(
                barmode="stack",
                xaxis_title="",
                yaxis_title="Esposizione",
                height=420,
                margin=dict(l=40, r=20, t=60, b=40),
                legend=dict(font=dict(size=12)),
            )
            fig_delta_stack.update_traces(textposition="inside")
            st.plotly_chart(fig_delta_stack, use_container_width=True)
    else:
        st.warning(f"Colonne mancanti per lo scostamento: {req_cols}")


with cluster_tab:
    # ----- Clustering -----
    st.subheader("Clustering su dati di dicembre")
    st.markdown(
        "**Nota sui cluster KMeans**: i cluster sono rinumerati dal piu' rischioso al meno rischioso usando l'Overdue medio. "
        "Cluster 0 = Overdue medio piu' alto (piu' rischio), i numeri aumentano al diminuire dell'Overdue medio. "
        "L'ordine delle feature scelte sopra non cambia questo ordinamento, che resta fisso per interpretazione."
    )
    num_cols = ensure_columns(df, ["Esposizione", "Overdue", "Variazione CRIF"])
    if not num_cols:
        st.warning("Nessuna colonna numerica disponibile per il clustering")
    else:
        features_sel = st.multiselect(
            "Seleziona feature per clustering",
            options=num_cols,
            default=[c for c in ["Esposizione", "Overdue"] if c in num_cols],
        )
        n_clusters = st.slider("Numero cluster", 2, 10, 3)

        df_cluster = df[features_sel].copy()
        df_cluster = df_cluster.fillna(df_cluster.median())

        scaler = StandardScaler()
        X = scaler.fit_transform(df_cluster)
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        labels = kmeans.fit_predict(X)
        sil = silhouette_score(X, labels)

        df_vis = df.copy()
        df_vis["Cluster_KMeans"] = labels

        # Riordina i cluster in modo consistente (0,1,2,...) ordinando per overdue medio decrescente
        order_ref = (
            df_vis.groupby("Cluster_KMeans")["Overdue"].mean().reset_index().sort_values("Overdue", ascending=False)
        )
        label_map = {row.Cluster_KMeans: int(idx) for idx, row in order_ref.reset_index(drop=True).iterrows()}
        df_vis["Cluster_KMeans"] = df_vis["Cluster_KMeans"].map(label_map)

        st.metric("Silhouette", f"{sil:.3f}")

        if len(features_sel) >= 2:
            fig_cluster = px.scatter(
                df_vis,
                x=features_sel[0],
                y=features_sel[1],
                color="Cluster_KMeans",
                hover_data=["CUSTOMER_CODE", "Brand"],
                title="Clustering KMeans",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            fig_cluster = px.histogram(
                df_vis,
                x=features_sel[0],
                color="Cluster_KMeans",
                nbins=40,
                title="Distribuzione per cluster",
            )
            st.plotly_chart(fig_cluster, use_container_width=True)

        # Due viste separate di supporto: una sull'esposizione e una sull'overdue
        viz_cols = st.columns(2)
        if "Esposizione" in df_vis.columns:
            with viz_cols[0]:
                fig_exp = px.histogram(
                    df_vis,
                    x="Esposizione",
                    color="Cluster_KMeans",
                    nbins=50,
                    title="Distribuzione esposizione per cluster",
                )
                st.plotly_chart(fig_exp, use_container_width=True)
        if "Overdue" in df_vis.columns:
            with viz_cols[1]:
                fig_od = px.histogram(
                    df_vis,
                    x="Overdue",
                    color="Cluster_KMeans",
                    nbins=50,
                    title="Distribuzione overdue per cluster",
                )
                st.plotly_chart(fig_od, use_container_width=True)

        total_expo_cluster = df_vis["Esposizione"].sum()
        cluster_summary = (
            df_vis.groupby("Cluster_KMeans")
            .agg(
                Numerosita=("CUSTOMER_CODE", "count"),
                Esposizione_media=("Esposizione", "mean"),
                Esposizione_mediana=("Esposizione", "median"),
                Esposizione_totale=("Esposizione", "sum"),
                Overdue_medio=("Overdue", "mean"),
                Overdue_mediana=("Overdue", "median"),
            )
            .reset_index()
            .sort_values("Cluster_KMeans")
        )
        if total_expo_cluster > 0:
            cluster_summary["Share_esposizione"] = cluster_summary["Esposizione_totale"] / total_expo_cluster

        cluster_summary_fmt = cluster_summary.copy()
        cluster_summary_fmt["Numerosita"] = cluster_summary_fmt["Numerosita"].map(lambda v: f"{v:,.0f}")
        for col in ["Esposizione_media", "Esposizione_mediana", "Esposizione_totale", "Overdue_medio", "Overdue_mediana"]:
            cluster_summary_fmt[col] = cluster_summary_fmt[col].map(lambda v: f"{v:,.0f}")
        if "Share_esposizione" in cluster_summary_fmt.columns:
            cluster_summary_fmt["Share_esposizione"] = cluster_summary_fmt["Share_esposizione"].map(lambda v: f"{v:.1%}")
        st.write("Sintesi cluster")
        st.dataframe(cluster_summary_fmt, use_container_width=True)

        # Heatmap quantili (p25, p50, p75) separate per Esposizione e Overdue
        quantiles = [0.25, 0.5, 0.75]
        for measure in ["Esposizione", "Overdue"]:
            q_series = df_vis.groupby("Cluster_KMeans")[measure].quantile(quantiles)
            rows = []
            for (cluster_id, q), val in q_series.items():
                rows.append({"Quantile": f"p{int(q*100)}", "Cluster": cluster_id, "Valore": val})
            quant_df = pd.DataFrame(rows)
            heat_data = quant_df.pivot(index="Quantile", columns="Cluster", values="Valore")
            fig_quant = px.imshow(
                heat_data,
                color_continuous_scale="Blues",
                aspect="auto",
                title=f"Quantili per cluster ({measure})",
            )
            st.plotly_chart(fig_quant, use_container_width=True)


with model_tab:
    # ----- Modello predittivo -----
    st.subheader("Modello predittivo (proxy aprile)")
    target_col = "CRIF Rating"
    id_cols = [c for c in ["CUSTOMER_CODE"] if c in df.columns]
    feature_cols = ensure_columns(
        df,
        [
            "CRIF Rating 2025-08",
            "Esposizione",
            "Overdue",
            "Variazione CRIF",
            "TIPOANAG",
            "Cluster Expo",
            "Brand",
        ],
    )

    if target_col not in df.columns or len(feature_cols) == 0:
        st.warning("Colonne insufficienti per il modello predittivo")
    else:
        df_model = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()
        df_model[target_col] = df_model[target_col].astype(str)

        categorical_cols = [c for c in feature_cols if df_model[c].dtype == "object"]
        numeric_cols = [c for c in feature_cols if c not in categorical_cols]

        # Cast categoriche a string per evitare mix di int/str
        for col in categorical_cols:
            df_model[col] = df_model[col].astype(str)

        # Cast numeriche a float per sicurezza
        for col in numeric_cols:
            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=
            [
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ],
            remainder="drop",
        )

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        X = df_model[feature_cols]
        y = df_model[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy test", f"{acc:.3f}")
        with col2:
            st.metric("F1 macro", f"{f1:.3f}")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).T
        st.write("Metriche per classe")
        st.dataframe(report_df, use_container_width=True)

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        fig_cm = px.imshow(
            cm,
            x=sorted(y.unique()),
            y=sorted(y.unique()),
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Predetto", "y": "Reale"},
            title="Confusion matrix"
        )
        fig_cm.update_layout(width=800, height=800, margin=dict(l=80, r=40, t=80, b=80))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Fit su tutto il set e predizioni proxy aprile
        pipe.fit(X, y)
        df_model_pred = df_model.copy()
        df_model_pred["Predizione_aprile"] = pipe.predict(X)
        # Aggiungi eventuali colonne ID (es. CUSTOMER_CODE) per visualizzazione
        if id_cols:
            df_model_pred[id_cols] = df.loc[df_model_pred.index, id_cols]
        # Delta rispetto al rating di dicembre: negativo migliora, positivo peggiora
        rating_order = ["1", "2", "3", "4", "5", "6", "7", "10", "Non valutabile", "Cessata", "New entry"]
        rank_map = {r: i for i, r in enumerate(rating_order)}
        df_model_pred["rank_dicembre"] = df_model_pred[target_col].map(rank_map)
        df_model_pred["rank_pred"] = df_model_pred["Predizione_aprile"].map(rank_map)
        df_model_pred["Delta_step"] = df_model_pred["rank_pred"] - df_model_pred["rank_dicembre"]

        def delta_label(val):
            if pd.isna(val):
                return "N/D"
            if val < 0:
                return "Migliora"
            if val > 0:
                return "Peggiora"
            return "Stabile"

        df_model_pred["Delta_vs_dicembre"] = df_model_pred["Delta_step"].apply(delta_label)
        # Sintesi aggregata delle predizioni
        pred_counts = (
            df_model_pred["Predizione_aprile"]
            .value_counts()
            .rename_axis("Predizione_aprile")
            .reset_index(name="Conteggio")
            .sort_values("Predizione_aprile")
        )
        pred_counts["Conteggio"] = pred_counts["Conteggio"].map(lambda v: f"{v:,.0f}")

        st.write("Distribuzione predizioni proxy aprile (dataset filtrato)")
        st.dataframe(pred_counts, use_container_width=True)

        # Delta aggregato vs dicembre (quante posizioni migliorano/peggiorano/stabili)
        delta_counts = (
            df_model_pred["Delta_vs_dicembre"]
            .value_counts()
            .reindex(["Migliora", "Stabile", "Peggiora", "N/D"])
            .dropna()
            .rename_axis("Delta_vs_dicembre")
            .reset_index(name="Conteggio")
        )
        delta_counts["Conteggio"] = delta_counts["Conteggio"].map(lambda v: f"{v:,.0f}")
        st.write("Delta predizione aprile vs rating dicembre")

        # Grafico di riepilogo delta (migliora/stabile/peggiora)
        delta_counts_num = (
            df_model_pred["Delta_vs_dicembre"]
            .value_counts()
            .reindex(["Migliora", "Stabile", "Peggiora", "N/D"])
            .dropna()
            .rename_axis("Delta_vs_dicembre")
            .reset_index(name="Conteggio")
        )
        fig_delta = px.bar(
            delta_counts_num,
            x="Delta_vs_dicembre",
            y="Conteggio",
            title="Delta predizione vs dicembre",
            text=delta_counts_num["Conteggio"].map(lambda v: f"{v:,.0f}"),
            color="Delta_vs_dicembre",
            color_discrete_map={"Migliora": "#2ecc40", "Stabile": "#7f8c8d", "Peggiora": "#ff4136", "N/D": "#999999"},
        )
        fig_delta.update_layout(showlegend=False, yaxis_title="Conteggio", xaxis_title="")
        fig_delta.update_traces(textposition="outside")
        st.plotly_chart(fig_delta, use_container_width=True)

        # Stacked bar per rating di dicembre: migliorano/stabili/peggiorano
        delta_by_rating = (
            df_model_pred.groupby([target_col, "Delta_vs_dicembre"])
            .size()
            .reset_index(name="Conteggio")
        )
        delta_by_rating["CRIF Rating"] = pd.Categorical(delta_by_rating[target_col], categories=rating_order, ordered=True)
        delta_by_rating = delta_by_rating.sort_values("CRIF Rating")

        fig_delta_stack = px.bar(
            delta_by_rating,
            x=target_col,
            y="Conteggio",
            color="Delta_vs_dicembre",
            title="Delta per rating di dicembre",
            color_discrete_map={"Migliora": "#2ecc40", "Stabile": "#7f8c8d", "Peggiora": "#ff4136", "N/D": "#999999"},
            text=delta_by_rating["Conteggio"].map(lambda v: f"{v:,.0f}"),
        )
        fig_delta_stack.update_layout(barmode="stack", xaxis_title="CRIF Rating dicembre", yaxis_title="Conteggio", legend_title="Delta")
        fig_delta_stack.update_traces(textposition="inside", textfont=dict(color="white"))
        st.plotly_chart(fig_delta_stack, use_container_width=True)

        st.write("Predizioni proxy aprile (tutte le righe del dataset filtrato)")
        cols_to_show = [c for c in ["CUSTOMER_CODE", target_col, "Predizione_aprile", "Delta_vs_dicembre", "Delta_step"] if c in df_model_pred.columns]
        st.dataframe(df_model_pred[cols_to_show], use_container_width=True)


with data_tab:
    # ----- Dati grezzi -----
    st.subheader("Dati grezzi")
    st.dataframe(df.head(200), use_container_width=True)
    st.caption("Prime 200 righe per riferimento rapido")
