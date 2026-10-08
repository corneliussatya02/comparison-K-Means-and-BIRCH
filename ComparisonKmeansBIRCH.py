import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

import warnings
import time
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans, Birch
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from io import BytesIO
from html import escape

st.set_page_config(page_title="Clustering K-Means & BIRCH", layout="wide")


def normalize_column_name(column_name):
    return ''.join(ch.lower() for ch in str(column_name) if ch.isalnum())


def find_matching_column(columns, keywords):
    normalized_keywords = [keyword.lower() for keyword in keywords]
    for column in columns:
        normalized_column = normalize_column_name(column)
        if all(keyword in normalized_column for keyword in normalized_keywords):
            return column
    return None


def format_rupiah(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return '-'

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(numeric_value):
        return '-'

    return f"Rp {numeric_value:,.0f}".replace(",", ".")


def build_cluster_purchase_tables(df, labels, column_specs, price_series=None, quantity_series=None):
    spec_map = {
        display_label: column_name
        for display_label, column_name in column_specs
        if column_name is not None and column_name in df.columns
    }

    bike_model_col = spec_map.get('Bike Model')
    store_location_col = spec_map.get('Store Location')
    gender_col = spec_map.get('Gender')

    if bike_model_col is None or store_location_col is None or gender_col is None:
        return pd.DataFrame(), pd.DataFrame()

    def format_gender_label(value):
        value_text = str(value).strip()
        if not value_text:
            return value_text
        if value_text.isupper() or value_text.islower():
            return value_text.title()
        return value_text

    labels_array = np.asarray(labels)
    cluster_ids = sorted(pd.unique(labels_array))

    gender_values_raw = df[gender_col].dropna().astype(str).str.strip().tolist()

    ordered_gender_values = []
    used_gender_keys = set()
    for preferred in ['female', 'male']:
        for gender_value in gender_values_raw:
            if gender_value.lower() == preferred and gender_value.lower() not in used_gender_keys:
                ordered_gender_values.append(gender_value)
                used_gender_keys.add(gender_value.lower())
                break
    for gender_value in gender_values_raw:
        if gender_value.lower() not in used_gender_keys:
            ordered_gender_values.append(gender_value)
            used_gender_keys.add(gender_value.lower())

    if not ordered_gender_values:
        ordered_gender_values = gender_values_raw

    gender_display_map = {gender_value: format_gender_label(gender_value) for gender_value in ordered_gender_values}

    summary_rows = []
    detail_rows = []

    for cluster_id in cluster_ids:
        cluster_mask = labels_array == cluster_id
        cluster_df = df.loc[cluster_mask].copy()

        bike_series = cluster_df[bike_model_col].dropna().astype(str).str.strip()
        bike_unique_values = bike_series.unique().tolist()

        # Hitung jumlah transaksi dan total quantity per bike model
        trx_by_bike = {}
        qty_by_bike = {}
        for bv in bike_unique_values:
            bv_mask = cluster_df[bike_model_col].astype(str).str.strip() == bv
            trx_by_bike[bv] = int(bv_mask.sum())
            if quantity_series is not None:
                qty_subset = quantity_series.loc[cluster_df.loc[bv_mask].index].dropna()
                qty_by_bike[bv] = int(qty_subset.sum()) if not qty_subset.empty else trx_by_bike[bv]
            else:
                qty_by_bike[bv] = trx_by_bike[bv]

        bike_counts = pd.DataFrame([
            {'Bike Model': bv, 'Jumlah Transaksi': trx_by_bike[bv], 'Total Quantity': qty_by_bike[bv]}
            for bv in bike_unique_values
        ])
        bike_counts = bike_counts.sort_values(['Total Quantity', 'Bike Model'], ascending=[False, True], kind='mergesort').reset_index(drop=True)

        for _, bike_row in bike_counts.iterrows():
            bike_value = str(bike_row['Bike Model']).strip()
            bike_trx = int(bike_row['Jumlah Transaksi'])
            bike_total_qty = int(bike_row['Total Quantity'])
            bike_mask = cluster_df[bike_model_col].astype(str).str.strip() == bike_value
            bike_df = cluster_df.loc[bike_mask].copy()

            price_value = '-'
            if price_series is not None and not bike_df.empty:
                price_subset = price_series.loc[bike_df.index].dropna()
                if not price_subset.empty:
                    price_mode = price_subset.mode()
                    price_value = price_mode.iloc[0] if not price_mode.empty else price_subset.iloc[0]

            store_series = bike_df[store_location_col].dropna().astype(str).str.strip()
            store_counts = store_series.value_counts().rename_axis('Store Location').reset_index(name='Jumlah Lokasi')
            store_counts = store_counts.sort_values(['Jumlah Lokasi', 'Store Location'], ascending=[False, True], kind='mergesort').reset_index(drop=True)

            top_store_value = str(store_counts.iloc[0]['Store Location']).strip() if not store_counts.empty else '-'
            top_store_count = int(store_counts.iloc[0]['Jumlah Lokasi']) if not store_counts.empty else 0

            # Tentukan gender terbanyak di seluruh bike model (untuk label Gender Terbanyak)
            model_gender_counts = bike_df[gender_col].dropna().astype(str).str.strip().value_counts()
            top_gender_raw = model_gender_counts.index[0] if not model_gender_counts.empty else '-'

            # Jumlah Gender = total Quantity yang dibeli gender terbanyak di lokasi terbanyak
            # Nilai ini berbeda dari Jumlah Lokasi (row count) karena qty per transaksi bisa > 1
            if top_store_value != '-' and quantity_series is not None:
                top_store_df = bike_df[
                    bike_df[store_location_col].astype(str).str.strip() == top_store_value
                ]
                gender_mask = top_store_df[gender_col].astype(str).str.strip() == top_gender_raw
                gender_qty_subset = quantity_series.loc[top_store_df[gender_mask].index].dropna()
                top_gender_count = int(gender_qty_subset.sum()) if not gender_qty_subset.empty else int(gender_mask.sum())
            elif top_store_value != '-':
                top_store_df = bike_df[
                    bike_df[store_location_col].astype(str).str.strip() == top_store_value
                ]
                gender_mask = top_store_df[gender_col].astype(str).str.strip() == top_gender_raw
                top_gender_count = int(gender_mask.sum())
            else:
                top_gender_count = int(model_gender_counts.iloc[0]) if not model_gender_counts.empty else 0

            summary_rows.append({
                'Cluster': f'Cluster {cluster_id}',
                'Rata-Rata Harga': format_rupiah(price_value),
                'Bike Model': bike_value,
                'Jumlah Transaksi': bike_trx,
                'Total Quantity': bike_total_qty,
                'Lokasi Terbanyak': top_store_value,
                'Jumlah Lokasi': top_store_count,
                'Gender Terbanyak': gender_display_map.get(top_gender_raw, top_gender_raw),
                'Jumlah Gender': top_gender_count,
                '_cluster_sort': cluster_id,
                '_bike_sort': -bike_total_qty,
            })

            for _, store_row in store_counts.iterrows():
                store_value = str(store_row['Store Location']).strip()
                store_total = int(store_row['Jumlah Lokasi'])
                store_mask = bike_df[store_location_col].astype(str).str.strip() == store_value
                store_df = bike_df.loc[store_mask].copy()
                store_gender_counts = store_df[gender_col].dropna().astype(str).str.strip().value_counts()

                # Total quantity per store
                if quantity_series is not None:
                    store_qty_subset = quantity_series.loc[store_df.index].dropna()
                    store_total_qty = int(store_qty_subset.sum()) if not store_qty_subset.empty else store_total
                else:
                    store_total_qty = store_total

                detail_row = {
                    'Cluster': f'Cluster {cluster_id}',
                    'Rata-Rata Harga': format_rupiah(price_value),
                    'Bike Model': bike_value,
                    'Jumlah Transaksi': bike_trx,
                    'Total Quantity': bike_total_qty,
                    'Store Location': store_value,
                    'Jumlah Lokasi': store_total,
                    'Total Quantity Lokasi': store_total_qty,
                    '_cluster_sort': cluster_id,
                    '_bike_sort': -bike_total_qty,
                    '_store_sort': -store_total,
                }

                # Hanya tampilkan gender terbanyak saja dengan jumlah quantity (sama seperti ringkasan)
                top_store_gender_raw = store_gender_counts.index[0] if not store_gender_counts.empty else '-'
                if top_store_gender_raw != '-' and quantity_series is not None:
                    store_gender_mask = store_df[gender_col].astype(str).str.strip() == top_store_gender_raw
                    store_gender_qty_subset = quantity_series.loc[store_df[store_gender_mask].index].dropna()
                    top_store_gender_count = int(store_gender_qty_subset.sum()) if not store_gender_qty_subset.empty else int(store_gender_mask.sum())
                else:
                    top_store_gender_count = int(store_gender_counts.iloc[0]) if not store_gender_counts.empty else 0

                pct = (top_store_gender_count / store_total_qty * 100) if store_total_qty > 0 else 100.0
                gender_str = gender_display_map.get(top_store_gender_raw, top_store_gender_raw) if top_store_gender_raw != '-' else '-'
                if gender_str != '-':
                    gender_str = f"{gender_str} ({pct:.0f}%)"

                detail_row['Gender Terbanyak'] = gender_str
                detail_row['Jumlah Gender'] = top_store_gender_count

                detail_rows.append(detail_row)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ['_cluster_sort', '_bike_sort', 'Bike Model'],
            ascending=[True, True, True],
            kind='mergesort'
        ).drop(columns=['_cluster_sort', '_bike_sort'])

    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            ['_cluster_sort', '_bike_sort', '_store_sort', 'Bike Model', 'Store Location'],
            ascending=[True, True, True, True, True],
            kind='mergesort'
        ).drop(columns=['_cluster_sort', '_bike_sort', '_store_sort'])

    ordered_gender_columns = [gender_display_map[gender_value] for gender_value in ordered_gender_values]

    ordered_summary_columns = [
        'Cluster',
        'Rata-Rata Harga',
        'Bike Model',
        'Jumlah Transaksi',
        'Total Quantity',
        'Lokasi Terbanyak',
        'Jumlah Lokasi',
        'Gender Terbanyak',
        'Jumlah Gender',
    ]
    ordered_summary_columns = [column for column in ordered_summary_columns if column in summary_df.columns]

    ordered_detail_columns = [
        'Cluster',
        'Rata-Rata Harga',
        'Bike Model',
        'Jumlah Transaksi',
        'Total Quantity',
        'Store Location',
        'Jumlah Lokasi',
        'Total Quantity Lokasi',
        'Gender Terbanyak',
        'Jumlah Gender',
    ]
    ordered_detail_columns = [column for column in ordered_detail_columns if column in detail_df.columns]

    if not summary_df.empty:
        summary_df = summary_df[ordered_summary_columns]
    if not detail_df.empty:
        detail_df = detail_df[ordered_detail_columns]

    return summary_df, detail_df

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

:root{
    --ink: #0f172a;
    --muted: #475569;
    --card-strong: rgba(255,255,255,0.92);
    --stroke: rgba(15, 23, 42, 0.10);
    --brand-1-rgb: 37, 99, 235;
    --brand-2-rgb: 124, 58, 237;
    --teal-rgb: 20, 184, 166;
    --rose-rgb: 244, 63, 94;
    --stroke-2: rgba(var(--brand-1-rgb), 0.18);
    --shadow: 0 14px 36px rgba(15, 23, 42, 0.10);
    --shadow-soft: 0 10px 24px rgba(15, 23, 42, 0.08);
    --brand-1: #2563eb; /* blue */
    --brand-2: #7c3aed; /* violet */
    --accent-teal: #14b8a6;
    --slate-1: #334155;
    --blackline: #111111;
    --upload-overlay-height: 222px;
    --glass-brown-rgb: 137, 98, 66;
    --glass-brown-bg: linear-gradient(135deg, rgba(255,248,241,0.32) 0%, rgba(196,156,122,0.18) 52%, rgba(133,98,66,0.14) 100%);
    --glass-brown-bg-strong: linear-gradient(135deg, rgba(255,248,241,0.42) 0%, rgba(196,156,122,0.24) 52%, rgba(133,98,66,0.18) 100%);
    --glass-brown-border: rgba(var(--glass-brown-rgb), 0.26);
    --glass-brown-shadow: 0 10px 24px rgba(97, 66, 42, 0.08);
    --glass-brown-focus: rgba(var(--glass-brown-rgb), 0.14);
    --glass-brown-text: #5f4632;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
    color: var(--ink);
}

.stMarkdown, .stText, .stCaption, .stSubheader, .stHeader, .stTitle {
    color: var(--ink);
}

.stMarkdown p {
    color: var(--muted);
}
            
.main-title {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: linear-gradient(135deg,
        #c49a6c 0%,
        #8b5e3c 25%,
        #1e3a8a 50%,
        #7c3aed 75%,
        #c49a6c 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    margin-bottom: 30px;
    padding: 24px 20px;
    line-height: 1.25;
    filter: drop-shadow(0 2px 12px rgba(124,58,237,0.15));
}

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

.main-title-wrap {
    position: relative;
    text-align: center;
    padding: 10px 0 6px;
    margin-bottom: 32px;
}

.main-title-wrap::before,
.main-title-wrap::after {
    content: '';
    display: block;
    margin: 0 auto;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, 
        transparent 0%, 
        #c49a6c 20%, 
        #7c3aed 50%, 
        #c49a6c 80%, 
        transparent 100%);
}

.main-title-wrap::before { margin-bottom: 18px; width: 60%; }
.main-title-wrap::after  { margin-top: 18px; width: 40%; }
            
.upload-box {
    min-height: 150px;
    padding: 22px 34px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    border-bottom: none;
    border-radius: 22px 22px 0 0;
    background: var(--glass-brown-bg-strong);
}

.upload-shell {
    width: 100%;
    margin: 0 0 6px;
}

.upload-card {
    border-radius: 22px;
    overflow: hidden;
    border: 1.5px solid var(--glass-brown-border);
    background: var(--glass-brown-bg);
    box-shadow: var(--glass-brown-shadow);
}

.upload-box h3 {
    margin: 0 0 10px 0;
    font-size: 28px;
    font-weight: 900;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    background: linear-gradient(135deg, 
        #c49a6c 0%, 
        #8b5e3c 30%, 
        #2563eb 60%, 
        #7c3aed 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
    filter: drop-shadow(0 2px 8px rgba(124,58,237,0.18));
}

.upload-box p {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: transparent;
    background: linear-gradient(90deg, #8b5e3c, #2563eb, #7c3aed);
    -webkit-background-clip: text;
    background-clip: text;
    opacity: 0.9;
}

.upload-bottom-bar {
    min-height: 50px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 14px 22px;
    background: rgba(255,255,255,0.07);
    border-top: 1px dashed rgba(147, 197, 253, 0.84);
}

.upload-meta {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.upload-meta-icon {
    width: 44px;
    height: 44px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    color: #94a3b8;
    background: rgba(241,245,249,0.08);
}

.upload-meta-text {
    min-width: 0;
}

.upload-meta-title {
    margin: 0;
    color: #334155;
    font-size: 16px;
    font-weight: 500;
}

.upload-meta-subtitle {
    margin: 4px 0 0;
    color: #64748b;
    font-size: 13px;
}

.upload-button-ghost {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 12px 20px;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.42);
    background: rgba(255,255,255,0.08);
    color: #0f172a;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
}

.upload-selected {
    width: 100%;
    margin: -6px 0 14px;
    display: flex;
    align-items: center;
    gap: 12px;
    color: #334155;
}

.upload-selected-name {
    font-size: 15px;
    font-weight: 500;
}

.upload-selected-size {
    margin-left: 8px;
    font-size: 13px;
    color: #64748b;
}

.section-title-wrapper {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 30px;
}

.section-title-wrapper hr {
    flex: 1;
    border: none;
    height: 2px;
    background: rgba(17, 17, 17, 0.32);
}

.section-title {
    margin: 0 15px;
    font-size: 22px;
    font-weight: 700;
    background: none;
    -webkit-background-clip: initial;
    -webkit-text-fill-color: var(--blackline);
    color: var(--blackline);
    font-family: 'Poppins', sans-serif;
}

.best-model-box {
    border-radius: 15px;
    padding: 25px;
    background: var(--glass-brown-bg-strong);
    border: 1.5px solid var(--glass-brown-border);
    border-left: 6px solid #8b5e3c;
    margin: 20px 0;
    box-shadow: var(--glass-brown-shadow);
}

.best-model-title {
    font-size: 20px;
    font-weight: 700;
    color: #5f4632;
    margin-bottom: 10px;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button {
    /* Softer, desaturated gradient to match page palette */
    background: linear-gradient(135deg, rgba(37,99,235,0.18) 0%, rgba(124,58,237,0.14) 65%, rgba(244,63,94,0.10) 100%) !important;
    color: #000000 !important;
    border: 1px solid rgba(15,23,42,0.12) !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.05rem !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 22px rgba(var(--brand-1-rgb), 0.12) !important;
    transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(var(--brand-1-rgb), 0.16) !important;
    border-color: rgba(15,23,42,0.18) !important;
    filter: saturate(1.02);
}
.stButton > button:active { transform: translateY(0px); }
.stDownloadButton > button {
    background: linear-gradient(135deg, 
        rgba(255,248,241,0.85) 0%, 
        rgba(196,156,122,0.55) 52%, 
        rgba(133,98,66,0.45) 100%) !important;
    color: var(--glass-brown-text) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    padding: 0.6rem 1.05rem !important;
    font-weight: 800 !important;
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: var(--glass-brown-shadow) !important;
    transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
}
.stDownloadButton > button:hover {
    background: linear-gradient(135deg, 
        rgba(255,248,241,0.95) 0%, 
        rgba(196,156,122,0.70) 52%, 
        rgba(133,98,66,0.60) 100%) !important;
    border-color: rgba(var(--glass-brown-rgb), 0.45) !important;
    color: var(--glass-brown-text) !important;
    transform: translateY(-1px);
}
.stDownloadButton > button:active { transform: translateY(0px); }

/* Inputs / selects */
div[data-baseweb="input"] {
    background: var(--glass-brown-bg) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--glass-brown-shadow) !important;
    overflow: hidden !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background: transparent !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: none !important;
    color: var(--glass-brown-text) !important;
}

div[data-baseweb="input"] button,
div[data-baseweb="input"] [role="button"] {
    background: rgba(var(--glass-brown-rgb), 0.10) !important;
    color: var(--glass-brown-text) !important;
    border-left: 1px solid rgba(var(--glass-brown-rgb), 0.16) !important;
}

div[data-baseweb="select"] > div {
    background: var(--glass-brown-bg) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--glass-brown-shadow) !important;
}

div[data-baseweb="select"] * {
    color: var(--glass-brown-text) !important;
}

div[data-baseweb="select"]:focus-within > div,
div[data-baseweb="input"]:focus-within,
div[data-baseweb="textarea"]:focus-within {
    box-shadow: 0 0 0 4px var(--glass-brown-focus), 0 12px 26px rgba(97,66,42,0.10) !important;
    border-color: rgba(var(--glass-brown-rgb), 0.34) !important;
}

/* Dropdown menu (options list) */
div[role="listbox"] {
    background: var(--glass-brown-bg-strong) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    box-shadow: 0 14px 30px rgba(97,66,42,0.12) !important;
    padding: 6px !important;
}
div[role="option"] {
    border-radius: 10px !important;
    color: var(--glass-brown-text) !important;
}
div[role="option"]:hover {
    background: rgba(var(--glass-brown-rgb), 0.10) !important;
}
div[role="option"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(255,248,241,0.36) 0%, rgba(var(--glass-brown-rgb), 0.20) 100%) !important;
}
span[data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(255,248,241,0.34) 0%, rgba(var(--glass-brown-rgb), 0.16) 100%) !important;
    border: 1px solid rgba(var(--glass-brown-rgb), 0.24) !important;
    border-radius: 999px !important;
}
span[data-baseweb="tag"] * {
    color: var(--glass-brown-text) !important;
    font-weight: 600 !important;
}

/* Dataframes: rounded, subtle border */
div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid #000000 !important;
    box-shadow: var(--shadow-soft);
    background: rgba(255,255,255,0.99);
}

/* Thicker gridlines (covers both table and grid renderers, depending on Streamlit version) */
div[data-testid="stDataFrame"] table {
    border-collapse: collapse !important;
    background: #ffffff !important;
    width: 100% !important;
}
div[data-testid="stDataFrame"] tbody tr {
    border-bottom: 5px solid #000000 !important;
}
div[data-testid="stDataFrame"] th,
div[data-testid="stDataFrame"] td {
    border: 5px solid #000000 !important;
    padding: 16px !important;
    font-size: 14px !important;
}
div[data-testid="stDataFrame"] th {
    background-color: #e8e8e8 !important;
    font-weight: 800 !important;
    border-width: 5px !important;
    color: #000000 !important;
}
div[data-testid="stDataFrame"] [role="grid"] {
    border: 5px solid #000000 !important;
}
div[data-testid="stDataFrame"] .ag-root-wrapper {
    border: 5px solid #000000 !important;
}
div[data-testid="stDataFrame"] .ag-header-cell,
div[data-testid="stDataFrame"] .ag-cell {
    border-right: 5px solid #000000 !important;
    border-bottom: 5px solid #000000 !important;
    padding: 14px !important;
}
div[data-testid="stDataFrame"] .ag-header-cell {
    background-color: #e8e8e8 !important;
    font-weight: 800 !important;
    color: #000000 !important;
}
div[data-testid="stDataFrame"] .ag-cell {
    background-color: #ffffff !important;
    color: #000000 !important;
}

/* Expanders */
div[data-testid="stExpander"] {
    border-radius: 14px;
    border: 1px solid var(--glass-brown-border);
    box-shadow: var(--glass-brown-shadow);
    background: var(--glass-brown-bg);
}

div[data-testid="stExpander"] summary {
    color: var(--glass-brown-text) !important;
}

div[data-testid="stExpander"] summary:hover {
    background: rgba(var(--glass-brown-rgb), 0.08) !important;
}

/* Alerts / info bars / status bars */
div[data-testid="stAlert"],
div[data-baseweb="notification"] {
    background: var(--glass-brown-bg-strong) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--glass-brown-shadow) !important;
    color: var(--glass-brown-text) !important;
}

div[data-testid="stAlert"] *,
div[data-baseweb="notification"] * {
    color: var(--glass-brown-text) !important;
}

div[data-testid="stAlert"] svg,
div[data-baseweb="notification"] svg {
    color: #8b6b4f !important;
    fill: #8b6b4f !important;
}

div[data-testid="stStatusWidget"] {
    border-radius: 14px !important;
    border: 1px solid var(--glass-brown-border) !important;
    background: var(--glass-brown-bg-strong) !important;
    box-shadow: var(--glass-brown-shadow) !important;
    overflow: hidden !important;
}

div[data-testid="stStatusWidget"] * {
    color: var(--glass-brown-text) !important;
}

div[data-testid="stStatusWidget"] summary,
div[data-testid="stStatusWidget"] [data-testid="stStatusWidgetHeader"] {
    background: rgba(var(--glass-brown-rgb), 0.08) !important;
}

div[data-testid="stStatusWidget"] summary:hover {
    background: rgba(var(--glass-brown-rgb), 0.12) !important;
}

/* Style file uploader untuk seamless dengan upload box */
[data-testid="stFileUploader"] {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    position: relative !important;
    top: calc(-1 * var(--upload-overlay-height)) !important;
    margin-bottom: calc(-1 * var(--upload-overlay-height) - 8px) !important;
    z-index: 4 !important;
}

[data-testid="stFileUploader"] > div,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div {
    width: 100% !important;
}

[data-testid="stFileUploader"] section {
    min-height: var(--upload-overlay-height) !important;
    opacity: 0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    min-height: var(--upload-overlay-height) !important;
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    cursor: pointer !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    min-height: var(--upload-overlay-height) !important;
    padding: 0 !important;
}

[data-testid="stFileUploaderFile"] {
    display: none !important;
}

@media (max-width: 768px) {
    :root{
        --upload-overlay-height: 274px;
    }

    .upload-box {
        min-height: 176px;
        padding: 22px 18px;
    }

    .upload-box h3 {
        font-size: 22px;
        margin-bottom: 12px;
    }

    .upload-bottom-bar {
        min-height: 98px;
        flex-direction: column;
        align-items: flex-start;
        padding: 14px 16px 16px;
    }

    .upload-button-ghost {
        width: 100%;
        justify-content: center;
    }

    [data-testid="stFileUploader"] {
        margin-bottom: calc(-1 * var(--upload-overlay-height) - 2px) !important;
    }
}

/* Best model typography */
.best-model-name {
    text-align: center;
    margin: 10px 0;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: 0.2px;
    background: linear-gradient(135deg, var(--accent-teal) 0%, var(--brand-1) 55%, var(--brand-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.best-model-divider {
    border: none;
    border-top: 1px solid rgba(var(--teal-rgb), 0.55);
    margin: 15px 0;
}
.best-model-metric {
    color: var(--muted);
    font-size: 14px;
    margin: 8px 0;
    text-align: center;
}
.best-model-metric strong { color: var(--ink); }
.best-model-metric .value {
    color: #8b5e3c;
    font-size: 16px;
    font-weight: 800;
}
            /* Styling untuk st.info / variance PCA */
div[data-testid="stAlert"][data-baseweb="notification"] {
    background: var(--glass-brown-bg-strong) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    box-shadow: var(--glass-brown-shadow) !important;
}

/* Tombol Proses Clustering - gradasi coklat */
.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, 
        rgba(255,248,241,0.85) 0%, 
        rgba(196,156,122,0.55) 52%, 
        rgba(133,98,66,0.45) 100%) !important;
    color: var(--glass-brown-text) !important;
    border: 1px solid var(--glass-brown-border) !important;
    border-radius: 14px !important;
    font-weight: 800 !important;
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 0.3px !important;
    box-shadow: var(--glass-brown-shadow) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, 
        rgba(255,248,241,0.95) 0%, 
        rgba(196,156,122,0.70) 52%, 
        rgba(133,98,66,0.60) 100%) !important;
    border-color: rgba(var(--glass-brown-rgb), 0.45) !important;
    color: var(--glass-brown-text) !important;
}
</style>
""", unsafe_allow_html=True)
import base64

try:
    with open('sepeda5.jpeg', 'rb') as _img:
        _b64 = base64.b64encode(_img.read()).decode()
except Exception:
    _b64 = ''

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpeg;base64,{_b64}") !important;
    background-size: cover !important;
    /* shift background image down so top isn't hidden behind Streamlit header */
    background-position: center calc(50% + 60px) !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}
.block-container {{ padding-top: 4.0rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title-wrap"><div class="main-title">PENGELOMPOKAN DATA PENJUALAN DENGAN K-MEANS DAN BIRCH</div></div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="upload-shell">
    <div class="upload-card">
        <div class="upload-box">
            <h3>Upload Dataset (CSV / Excel)</h3>
            <p>Drag & drop atau klik untuk memilih file CSV/Excel</p>
        </div>
        <div class="upload-bottom-bar">
            <div class="upload-meta">
                <div class="upload-meta-icon">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                        <path d="M7 18a4 4 0 0 1-.4-7.98A6 6 0 0 1 18.9 9A4.5 4.5 0 1 1 18.5 18H14" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M12 15V8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                        <path d="M9.5 10.5L12 8l2.5 2.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                <div class="upload-meta-text">
                    <p class="upload-meta-title">Drag and drop file here</p>
                    <p class="upload-meta-subtitle">Limit 200MB per file &bull; CSV, XLSX</p>
                </div>
            </div>
            <div class="upload-button-ghost">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <path d="M12 16V8" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <path d="M8.5 11.5L12 8l3.5 3.5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M5 19h14" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <span>Upload</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload dataset", type=["csv", "xlsx"], label_visibility="collapsed")

if uploaded_file is not None:
    uploaded_size_mb = uploaded_file.size / (1024 * 1024)
    uploaded_file_name = escape(uploaded_file.name)
    st.markdown(f"""
                <div class="upload-selected">
                <div class="upload-meta-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M7 3.75h6.25L18.25 8.75V20.25H7V3.75Z" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
                <path d="M13 3.75V9H18.25" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/>
            </svg>
        </div>
        <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
            <span class="upload-selected-name">{uploaded_file_name}</span>
            <span class="upload-selected-size">{uploaded_size_mb:.1f}MB</span>
            <span style="
                font-size:11px;
                font-weight:700;
                color:#166534;
                background:#dcfce7;
                border:1px solid #86efac;
                border-radius:999px;
                padding:2px 10px;
                letter-spacing:0.3px;
            ">✓ Upload Data Berhasil!</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # --- Preview raw uploaded data (before preprocessing) ---
    st.subheader("Preview Data")
    df_preview = df.copy()
    numeric_cols_preview = df_preview.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols_preview and len(df_preview) <= 20000:
        df_preview[numeric_cols_preview] = df_preview[numeric_cols_preview].round(3)
    st.dataframe(df_preview, use_container_width=True, height=360)

    # --- Automatic preprocessing (hidden UI) ---
    # Hapus missing value
    df = df.dropna()
    df_raw_before_currency = df.copy()

    # Konversi harga (tampilkan pilihan kolom dan kurs, tetap tampilkan hasil konversi)
    currency_cols = st.multiselect(
        "Pilih kolom harga USD yang ingin dikonversi ke Rupiah:",
        df.select_dtypes(include=np.number).columns.tolist()
    )
    kurs = st.number_input("Masukkan Kurs Dollar ke Rupiah", value=16500)
    price_display_col = find_matching_column(df.columns, ('price',))
    if price_display_col is None:
        price_display_col = find_matching_column(df.columns, ('harga',))
    if price_display_col is not None and price_display_col not in currency_cols:
        currency_cols = [price_display_col, *currency_cols]
    currency_cols = list(dict.fromkeys(currency_cols))
    if currency_cols:
        for col in currency_cols:
            df[col] = df[col] * kurs
        st.success("Konversi USD -> Rupiah berhasil")
    df_raw_after_currency = df.copy()
    price_display_series = df_raw_after_currency[price_display_col].copy() if price_display_col is not None else None

    # Standarisasi akan dilakukan SETELAH Label Encoding
    # agar semua atribut (numerik + encoded) distandarisasi sekaligus

    # Mapping Store_Location ke State (nama negara bagian)
    state_mapping = {
        "Houston":      "Texas",
        "San Antonio":  "Texas",
        "Phoenix":      "Arizona",
        "Los Angeles":  "California",
        "New York":     "New York",
        "Chicago":      "Illinois",
        "Philadelphia": "Pennsylvania",
    }
    _store_loc_col = find_matching_column(df.columns, ('store', 'location'))
    if _store_loc_col is not None:
        df["State"] = df[_store_loc_col].map(state_mapping)
        # Jika ada nilai yang tidak terpetakan, isi dengan nilai asli agar tidak NaN
        df["State"] = df["State"].fillna(df[_store_loc_col])

    # Encode kolom string secara otomatis dan simpan label encoders
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    label_encoders = {}
    if string_cols:
        for col in string_cols:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Simpan backup integer encoded SEBELUM standarisasi, untuk keperluan inverse_transform
    encoded_cols_backup = [f'{col}_encoded' for col in string_cols if f'{col}_encoded' in df.columns]
    df_encoded_int_backup = df[encoded_cols_backup].astype(int).copy() if encoded_cols_backup else pd.DataFrame()

    # Standarisasi semua atribut (numerik asli + kolom _encoded hasil LabelEncoder)
    all_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if all_numeric_cols:
        scaler_preprocessing = StandardScaler()
        df[all_numeric_cols] = scaler_preprocessing.fit_transform(df[all_numeric_cols])

        # Tampilkan mappings untuk referensi (user ingin melihat mapping)
    with st.expander("Lihat Perubahan Data Kategori", expanded=True):
        cols_pairs = [string_cols[i:i+3] for i in range(0, len(string_cols), 3)]
        for pair in cols_pairs:
            grid_cols = st.columns(len(pair))
            for idx, col in enumerate(pair):
                with grid_cols[idx]:
                    st.markdown(f"##### {col}")
                    mapping_dict = dict(zip(
                        label_encoders[col].classes_,
                        label_encoders[col].transform(label_encoders[col].classes_)
                    ))
                    mapping_df = pd.DataFrame(
                        list(mapping_dict.items()),
                        columns=['Nilai Asli', 'Kode']
                 )
                    row_count = len(mapping_df)
                    row_height = 35
                    header_height = 38
                    padding = 10
                    max_height = 320  # batas maksimum tinggi tabel

                    dynamic_height = min(header_height + (row_count * row_height) + padding, max_height)
                    st.dataframe(
                        mapping_df,
                        use_container_width=True,
                        hide_index=True,
                        height=dynamic_height
                    )

    st.markdown("---")
    st.markdown("### Preview setelah preprocessing")
    df_display = df.copy()
    float_cols_display = df_display.select_dtypes(include=["float", "float64", "float32"]).columns.tolist()
    column_cfg_display = {
        col: st.column_config.NumberColumn(format="%.3f") for col in float_cols_display
    }
    st.dataframe(
        df_display,
        use_container_width=True,
        height=360,
        column_config=column_cfg_display,
    )

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Dataset minimal memiliki 2 kolom numerik.")
    else:

        selected_features = st.multiselect(
            "Pilih atribut untuk clustering:",
            numeric_cols,
            default=numeric_cols[:2]
        )

        if len(selected_features) < 2:
            st.warning("Pilih minimal 2 atribut.")
        else:

            k = st.number_input("Jumlah Cluster (k)", min_value=2, max_value=10, value=3)

            if st.button("Proses Clustering"):

                X = df[selected_features]

                if len(X) < k:
                    st.error("Jumlah data lebih kecil dari jumlah cluster.")
                    st.stop()

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                max_components = min(3, X_scaled.shape[1])
                pca = PCA(n_components=max_components)
                X_pca = pca.fit_transform(X_scaled)

                explained_var = pca.explained_variance_ratio_

                for i, var in enumerate(explained_var):
                    st.info(f"Variance PCA {i+1}: {var:.2%}")

                # OPTIMIZED CLUSTERING MODELS FOR BIG DATA
                models = {}
                scores = {}
                processing_times = {}

                # K-MEANS (Random init with max_iter=500)
                with st.status("Pemrosesan K-Means...", expanded=False) as status_kmeans:
                    start_time = time.time()
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, 
                                   max_iter=100, algorithm='lloyd', init='random', tol=1e-5)
                    kmeans_labels = kmeans.fit_predict(X_scaled)
                    kmeans_time = time.time() - start_time
                    
                    models['K-Means'] = kmeans_labels
                    scores['K-Means'] = {
                        'Silhouette': silhouette_score(X_scaled, kmeans_labels),
                        'Davies-Bouldin': davies_bouldin_score(X_scaled, kmeans_labels)
                    }
                    processing_times['K-Means'] = kmeans_time
                    status_kmeans.update(label="Proses K-Means Sukses", state="complete")

                # BIRCH (threshold=2.5)
                with st.status("Pemrosesan BIRCH...", expanded=False) as status_birch:
                    start_time = time.time()
                    birch = Birch(n_clusters=k, threshold=2.5, branching_factor=50)
                    birch_labels = birch.fit_predict(X_scaled)
                    birch_time = time.time() - start_time
                    
                    models['BIRCH'] = birch_labels
                    scores['BIRCH'] = {
                        'Silhouette': silhouette_score(X_scaled, birch_labels),
                        'Davies-Bouldin': davies_bouldin_score(X_scaled, birch_labels)
                    }
                    processing_times['BIRCH'] = birch_time
                    status_birch.update(label="Proses BIRCH Sukses", state="complete")

                # FIND BEST MODEL - Berdasarkan 3 Metrik: Silhouette, DBI, dan Processing Time
                best_model_name = None
                best_score = -1
                model_rankings = {}
                
                # Normalisasi skor untuk setiap metrik
                sil_scores = [scores[m]['Silhouette'] for m in scores.keys()]
                dbi_scores = [scores[m]['Davies-Bouldin'] for m in scores.keys()]
                time_scores = [processing_times[m] for m in scores.keys()]
                
                sil_min, sil_max = min(sil_scores), max(sil_scores)
                dbi_min, dbi_max = min(dbi_scores), max(dbi_scores)
                time_min, time_max = min(time_scores), max(time_scores)
                
                # Normalisasi ke range 0-1 (lebih tinggi = lebih baik)
                for model_name in scores.keys():
                    sil_normalized = (scores[model_name]['Silhouette'] - sil_min) / (sil_max - sil_min) if sil_max > sil_min else 0.5
                    dbi_normalized = 1 - ((scores[model_name]['Davies-Bouldin'] - dbi_min) / (dbi_max - dbi_min)) if dbi_max > dbi_min else 0.5
                    time_normalized = 1 - ((processing_times[model_name] - time_min) / (time_max - time_min)) if time_max > time_min else 0.5
                    
                    # Bobot: Silhouette 40%, DBI 40%, Processing Time 20%
                    combined_score = (sil_normalized * 0.4) + (dbi_normalized * 0.4) + (time_normalized * 0.2)
                    model_rankings[model_name] = {
                        'silhouette_norm': sil_normalized,
                        'dbi_norm': dbi_normalized,
                        'time_norm': time_normalized,
                        'combined': combined_score
                    }
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_model_name = model_name

                best_labels = models[best_model_name]
                best_sil = scores[best_model_name]['Silhouette']
                best_dbi = scores[best_model_name]['Davies-Bouldin']

                st.markdown("""
                <div class="section-title-wrapper">
                    <hr>
                    <div class="section-title">HASIL CLUSTERING</div>
                    <hr>
                </div>
                """, unsafe_allow_html=True)

                # Display Best Model
                st.markdown(f"""
                <div class="best-model-box">
                    <div class="best-model-title">ALGORITMA TERBAIK</div>
                    <div class="best-model-name">{best_model_name}</div>
                    <hr class="best-model-divider">
                    <p class="best-model-metric"><strong>Silhouette Score:</strong> <span class="value">{best_sil:.4f}</span></p>
                    <p class="best-model-metric"><strong>Davies-Bouldin Index:</strong> <span class="value">{best_dbi:.4f}</span></p>
                    <p class="best-model-metric"><strong>Processing Time:</strong> <span class="value">{processing_times[best_model_name]:.3f} s</span></p>
                </div>
                """, unsafe_allow_html=True)

                # Show all model comparisons dengan informasi lebih detail
                st.markdown("### Perbandingan Model")
                
                # Urutkan model dengan terbaik di atas
                sorted_models = sorted(scores.keys(), key=lambda m: model_rankings[m]['combined'], reverse=True)
                comparison_data = []
                for model_name in sorted_models:
                    status = 'TERBAIK' if model_name == best_model_name else ''
                    comparison_data.append({
                        'Model': model_name,
                        'Silhouette Score': f"{scores[model_name]['Silhouette']:.4f}",
                        'Davies-Bouldin Index': f"{scores[model_name]['Davies-Bouldin']:.4f}",
                        'Processing Time (s)': f"{processing_times[model_name]:.3f}",
                        'Status': status
                    })
                comparison_df = pd.DataFrame(comparison_data)
                
                # Tampilkan dengan styling
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)

                st.subheader("Visualisasi PCA 2D - Perbandingan K-Means vs BIRCH")

                col_2d_1, col_2d_2 = st.columns(2)
                xlab_2d = f"PCA 1 ({explained_var[0]*100:.1f}%)"
                ylab_2d = f"PCA 2 ({explained_var[1]*100:.1f}%)"
                cluster_cmap = plt.get_cmap('tab10', k)
                cluster_norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cluster_cmap.N)

                with col_2d_1:
                    fig_km, ax_km = plt.subplots(figsize=(8.6, 6.6), dpi=140, facecolor='#fbfbfe')
                    kmeans_labels = models['K-Means']
                    scatter_km = ax_km.scatter(
                        X_pca[:, 0], X_pca[:, 1],
                        c=kmeans_labels, cmap=cluster_cmap, norm=cluster_norm,
                        s=120, alpha=0.92, edgecolors='#ffffff', linewidth=0.85
                    )
                    ax_km.set_title(
                        f"K-Means\nSilhouette: {scores['K-Means']['Silhouette']:.4f} | DBI: {scores['K-Means']['Davies-Bouldin']:.4f}",
                        fontsize=14, fontweight='bold', color='#2563eb', pad=16
                    )
                    ax_km.set_xlabel(xlab_2d, fontsize=13, fontweight='600', color='#2563eb')
                    ax_km.set_ylabel(ylab_2d, fontsize=13, fontweight='600', color='#2563eb')
                    ax_km.tick_params(axis='both', labelsize=11, colors='#334155')
                    ax_km.grid(True, linestyle='--', linewidth=1.0, alpha=0.35, color='#2563eb')
                    ax_km.axhline(0, linewidth=1.5, alpha=0.15, color='#0f172a')
                    ax_km.axvline(0, linewidth=1.5, alpha=0.15, color='#0f172a')
                    ax_km.set_facecolor('#fbfbfe')
                    ax_km.set_aspect('equal', adjustable='box')
                    for spine in ax_km.spines.values():
                        spine.set_linewidth(1.3)
                        spine.set_alpha(0.3)
                        spine.set_color('#2563eb')
                    cbar = fig_km.colorbar(scatter_km, ax=ax_km, pad=0.02)
                    cbar.set_label('Cluster', color='#334155', fontsize=12, fontweight='600')
                    if k <= 10:
                        cbar.set_ticks(list(range(k)))
                    cbar.outline.set_linewidth(1.3)
                    cbar.outline.set_edgecolor('#2563eb')
                    fig_km.tight_layout()
                    st.pyplot(fig_km)

                with col_2d_2:
                    fig_birch, ax_birch = plt.subplots(figsize=(8.6, 6.6), dpi=140, facecolor='#fbfbfe')
                    birch_labels = models['BIRCH']
                    scatter_birch = ax_birch.scatter(
                        X_pca[:, 0], X_pca[:, 1],
                        c=birch_labels, cmap=cluster_cmap, norm=cluster_norm,
                        s=120, alpha=0.92, edgecolors='#ffffff', linewidth=0.85
                    )
                    ax_birch.set_title(
                        f"BIRCH\nSilhouette: {scores['BIRCH']['Silhouette']:.4f} | DBI: {scores['BIRCH']['Davies-Bouldin']:.4f}",
                        fontsize=14, fontweight='bold', color='#7c3aed', pad=16
                    )
                    ax_birch.set_xlabel(xlab_2d, fontsize=13, fontweight='600', color='#7c3aed')
                    ax_birch.set_ylabel(ylab_2d, fontsize=13, fontweight='600', color='#7c3aed')
                    ax_birch.tick_params(axis='both', labelsize=11, colors='#334155')
                    ax_birch.grid(True, linestyle='--', linewidth=1.0, alpha=0.35, color='#7c3aed')
                    ax_birch.axhline(0, linewidth=1.5, alpha=0.15, color='#0f172a')
                    ax_birch.axvline(0, linewidth=1.5, alpha=0.15, color='#0f172a')
                    ax_birch.set_facecolor('#fbfbfe')
                    ax_birch.set_aspect('equal', adjustable='box')
                    for spine in ax_birch.spines.values():
                        spine.set_linewidth(1.3)
                        spine.set_alpha(0.3)
                        spine.set_color('#7c3aed')
                    cbar = fig_birch.colorbar(scatter_birch, ax=ax_birch, pad=0.02)
                    cbar.set_label('Cluster', color='#334155', fontsize=12, fontweight='600')
                    if k <= 10:
                        cbar.set_ticks(list(range(k)))
                    cbar.outline.set_linewidth(1.3)
                    cbar.outline.set_edgecolor('#7c3aed')
                    fig_birch.tight_layout()
                    st.pyplot(fig_birch)

                if X_pca.shape[1] >= 3:
                    st.subheader("Visualisasi PCA 3D - Perbandingan K-Means vs BIRCH")

                    col_3d_1, col_3d_2 = st.columns(2)
                    xlab_3d = f"PCA-1, {explained_var[0]*100:.2f}% "
                    ylab_3d = f"PCA-2, {explained_var[1]*100:.2f}% "
                    zlab_3d = f"PCA-3, {explained_var[2]*100:.2f}% "
                    markers_3d = ['o', 'x', '^', 's', 'D', 'P', '*', 'v', '<', '>']

                    with col_3d_1:
                        fig_km_3d = plt.figure(figsize=(11, 8), dpi=140, facecolor='#fdf8f4')
                        ax_km_3d = fig_km_3d.add_subplot(111, projection='3d')
                        kmeans_labels = models['K-Means']

                        title_color = '#8b5e3c'
                        title_text = "K-Means (3D)"
                        if best_model_name == 'K-Means':
                            title_text = "K-Means (3D) TERBAIK"
                            title_color = '#059669'

                        markers_3d = ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o']
                        np.random.seed(42)
                        X_pca_jitter = X_pca.copy()
                        jitter_scale = (X_pca[:, 2].max() - X_pca[:, 2].min()) * 0.08
                        X_pca_jitter[:, 2] += np.random.normal(0, jitter_scale, size=X_pca.shape[0])

                        for cluster_id in range(k):
                            mask = kmeans_labels == cluster_id
                            if not np.any(mask): continue
                            ax_km_3d.scatter(
                                X_pca_jitter[mask, 0], X_pca_jitter[mask, 1], X_pca_jitter[mask, 2],
                                c=[cluster_cmap(cluster_id)],
                                s=32, alpha=0.70,
                                edgecolors='white', linewidths=0.4,
                                depthshade=True,
                                marker='o',
                                label=f"Cluster {cluster_id}"
                            )

                        ax_km_3d.set_title(
                            f"{title_text}\nSilhouette: {scores['K-Means']['Silhouette']:.4f}  |  DBI: {scores['K-Means']['Davies-Bouldin']:.4f}  |  Waktu: {processing_times['K-Means']:.3f}s",
                            fontsize=12, fontweight='bold', color=title_color, pad=18
                        )
                        ax_km_3d.set_xlabel(xlab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_km_3d.set_ylabel(ylab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_km_3d.set_zlabel(zlab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_km_3d.tick_params(colors='#5f4632', labelsize=8)
                        ax_km_3d.xaxis.pane.fill = True
                        ax_km_3d.yaxis.pane.fill = True
                        ax_km_3d.zaxis.pane.fill = True
                        ax_km_3d.xaxis.pane.set_facecolor((1.0, 0.97, 0.93, 0.35))
                        ax_km_3d.yaxis.pane.set_facecolor((0.98, 0.94, 0.89, 0.25))
                        ax_km_3d.zaxis.pane.set_facecolor((0.96, 0.91, 0.85, 0.18))
                        ax_km_3d.xaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_km_3d.yaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_km_3d.zaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_km_3d.grid(True, alpha=0.30, linewidth=0.6, color='#c49a6c')
                        ax_km_3d.view_init(elev=20, azim=-55)
                        ax_km_3d.legend(loc='upper left', fontsize=9, frameon=True,
                                        framealpha=0.88, facecolor='#fff8f1', edgecolor='#c49a6c')
                        fig_km_3d.tight_layout()
                        st.pyplot(fig_km_3d)

                    with col_3d_2:
                        fig_birch_3d = plt.figure(figsize=(11, 8), dpi=140, facecolor='#fdf8f4')
                        ax_birch_3d = fig_birch_3d.add_subplot(111, projection='3d')
                        birch_labels = models['BIRCH']

                        title_color = '#7c3aed'
                        title_text = "BIRCH (3D)"
                        if best_model_name == 'BIRCH':
                            title_text = "BIRCH (3D) TERBAIK"
                            title_color = '#059669'

                        np.random.seed(42)
                        X_pca_jitter2 = X_pca.copy()
                        X_pca_jitter2[:, 2] += np.random.normal(0, jitter_scale, size=X_pca.shape[0])

                        for cluster_id in range(k):
                            mask = birch_labels == cluster_id
                            if not np.any(mask): continue
                            ax_birch_3d.scatter(
                                X_pca_jitter2[mask, 0], X_pca_jitter2[mask, 1], X_pca_jitter2[mask, 2],
                                c=[cluster_cmap(cluster_id)],
                                s=32, alpha=0.70,
                                edgecolors='white', linewidths=0.4,
                                depthshade=True,
                                marker='o',
                                label=f"Cluster {cluster_id}"
                            )

                        ax_birch_3d.set_title(
                            f"{title_text}\nSilhouette: {scores['BIRCH']['Silhouette']:.4f}  |  DBI: {scores['BIRCH']['Davies-Bouldin']:.4f}  |  Waktu: {processing_times['BIRCH']:.3f}s",
                            fontsize=12, fontweight='bold', color=title_color, pad=18
                        )
                        ax_birch_3d.set_xlabel(xlab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_birch_3d.set_ylabel(ylab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_birch_3d.set_zlabel(zlab_3d, fontsize=10, color='#8b5e3c', labelpad=12)
                        ax_birch_3d.tick_params(colors='#5f4632', labelsize=8)
                        ax_birch_3d.xaxis.pane.fill = True
                        ax_birch_3d.yaxis.pane.fill = True
                        ax_birch_3d.zaxis.pane.fill = True
                        ax_birch_3d.xaxis.pane.set_facecolor((1.0, 0.97, 0.93, 0.35))
                        ax_birch_3d.yaxis.pane.set_facecolor((0.98, 0.94, 0.89, 0.25))
                        ax_birch_3d.zaxis.pane.set_facecolor((0.96, 0.91, 0.85, 0.18))
                        ax_birch_3d.xaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_birch_3d.yaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_birch_3d.zaxis.pane.set_edgecolor((0.55, 0.38, 0.26, 0.25))
                        ax_birch_3d.grid(True, alpha=0.30, linewidth=0.6, color='#c49a6c')
                        ax_birch_3d.view_init(elev=20, azim=-55)
                        ax_birch_3d.legend(loc='upper left', fontsize=9, frameon=True,
                                           framealpha=0.88, facecolor='#fff8f1', edgecolor='#c49a6c')
                        fig_birch_3d.tight_layout()
                        st.pyplot(fig_birch_3d)

                # Ganti nama kolom dengan nama algoritma yang dipilih
                cluster_column_name = f"Cluster_{best_model_name}"
                df[cluster_column_name] = best_labels

                cluster_attribute_specs = [
                    ('Bike Model', find_matching_column(df.columns, ('bike', 'model'))),
                    ('Store Location', find_matching_column(df.columns, ('store', 'location'))),
                    ('Gender', find_matching_column(df.columns, ('gender',))),
                ]
                special_display_columns = [
                    column_name
                    for _, column_name in cluster_attribute_specs
                    if column_name is not None
                ]
                special_display_columns = list(dict.fromkeys(special_display_columns))

                # Decode kolom string yang dipilih untuk ditampilkan di preview
                # Gunakan df_encoded_int_backup (integer asli sebelum standarisasi)
                display_features = []
                for feature in selected_features:
                    original_col = feature.replace('_encoded', '')

                    if original_col in label_encoders and feature == f'{original_col}_encoded':
                        if feature in df_encoded_int_backup.columns:
                            df[original_col] = label_encoders[original_col].inverse_transform(
                                df_encoded_int_backup[feature].values
                            )
                        else:
                            # fallback: clip & round nilai z-score ke indeks valid
                            n_classes = len(label_encoders[original_col].classes_)
                            safe_idx = df[feature].round().astype(int).clip(0, n_classes - 1)
                            df[original_col] = label_encoders[original_col].inverse_transform(safe_idx)
                        display_features.append(original_col)
                    else:
                        display_features.append(feature)

                display_features = list(dict.fromkeys(display_features + special_display_columns))

                # Export moved to the end of processin  g (download button will appear after all results)

                st.subheader("Analisis Frekuensi Cluster K-Means vs BIRCH")
                
                col_freq1, col_freq2 = st.columns(2)
                
                with col_freq1:
                    st.markdown("**K-Means - Frekuensi Cluster**")
                    kmeans_freq = pd.Series(models['K-Means']).value_counts().sort_index()
                    
                    fig_kmeans_freq, ax_kmeans_freq = plt.subplots(figsize=(8, 4), facecolor='white')
                    kmeans_freq.plot(kind='bar', ax=ax_kmeans_freq, color='#2563eb', edgecolor='#7c3aed', linewidth=1.5)
                    ax_kmeans_freq.set_title(f"Jumlah Data per Cluster (K-Means)", fontsize=12, fontweight='bold', color='#2563eb')
                    ax_kmeans_freq.set_xlabel("Cluster", color='#2563eb')
                    ax_kmeans_freq.set_ylabel("Jumlah Data", color='#2563eb')
                    ax_kmeans_freq.grid(axis='y', alpha=0.3, color='#2563eb')
                    ax_kmeans_freq.tick_params(colors='#2563eb')
                    ax_kmeans_freq.set_facecolor('#ffffff')
                    plt.tight_layout()
                    st.pyplot(fig_kmeans_freq)
                    
                    # Hitung total quantity per cluster K-Means
                    _qty_col_freq = find_matching_column(df_raw_after_currency.columns, ('quantity',)) or find_matching_column(df_raw_after_currency.columns, ('qty',))
                    _bike_col_freq = find_matching_column(df_raw_after_currency.columns, ('bike', 'model'))
                    _store_col_freq = find_matching_column(df_raw_after_currency.columns, ('store', 'location'))
                    _gender_col_freq = find_matching_column(df_raw_after_currency.columns, ('gender',))
                    _price_col_freq = find_matching_column(df_raw_after_currency.columns, ('price',)) or find_matching_column(df_raw_after_currency.columns, ('harga',))

                    def _build_freq_rows(model_labels, model_freq):
                        rows = []
                        for _cid in model_freq.index:
                            _cmask = np.asarray(model_labels) == _cid
                            _cidx = df.index[_cmask]
                            _raw_sub = df_raw_after_currency.loc[_cidx]
                            # Total quantity
                            _cqty = int(_raw_sub[_qty_col_freq].dropna().sum()) if _qty_col_freq else int(_cmask.sum())
                            
                            # Harga Dominan (rentang harga paling sering muncul)
                            _segmen_harga = '-'
                            if _price_col_freq and not _raw_sub[_price_col_freq].dropna().empty:
                                _cprice_series = _raw_sub[_price_col_freq].dropna()
                                if _cprice_series.nunique() > 1:
                                    try:
                                        # Membagi range harga ke dalam 4 bins dan mencari interval frekuensi tertinggi
                                        _price_bins = _cprice_series.value_counts(bins=4)
                                        _top_interval = _price_bins.index[0]
                                        _segmen_harga = f"{format_rupiah(_top_interval.left)} - {format_rupiah(_top_interval.right)}"
                                    except Exception:
                                        _segmen_harga = format_rupiah(_cprice_series.mode().iloc[0])
                                else:
                                    _segmen_harga = format_rupiah(_cprice_series.iloc[0])
                                    
                            # Jenis sepeda terbanyak
                            _bike_top = _raw_sub[_bike_col_freq].dropna().mode().iloc[0] if _bike_col_freq and not _raw_sub[_bike_col_freq].dropna().empty else '-'
                            # Lokasi terbanyak
                            _loc_top = _raw_sub[_store_col_freq].dropna().mode().iloc[0] if _store_col_freq and not _raw_sub[_store_col_freq].dropna().empty else '-'
                            # Gender terbanyak
                            _gen_top = _raw_sub[_gender_col_freq].dropna().mode().iloc[0] if _gender_col_freq and not _raw_sub[_gender_col_freq].dropna().empty else '-'
                            rows.append({
                                'Cluster': _cid,
                                'Jumlah Transaksi': int(_cmask.sum()),
                                'Total Quantity': _cqty,
                                'Harga Dominan': _segmen_harga,
                                'Jenis Sepeda': _bike_top,
                                'Lokasi': _loc_top,
                                'Gender': _gen_top,
                                'Persentase': f"{int(_cmask.sum()) / len(model_labels) * 100:.3f}%"
                            })
                        return pd.DataFrame(rows)

                    kmeans_freq_df = _build_freq_rows(models['K-Means'], kmeans_freq)
                    st.dataframe(kmeans_freq_df, use_container_width=True, hide_index=True)
                
                with col_freq2:
                    st.markdown("**BIRCH - Frekuensi Cluster**")
                    birch_freq = pd.Series(models['BIRCH']).value_counts().sort_index()
                    
                    fig_birch_freq, ax_birch_freq = plt.subplots(figsize=(8, 4), facecolor='white')
                    birch_freq.plot(kind='bar', ax=ax_birch_freq, color='#7c3aed', edgecolor='#2563eb', linewidth=1.5)
                    ax_birch_freq.set_title(f"Jumlah Data per Cluster (BIRCH)", fontsize=12, fontweight='bold', color='#7c3aed')
                    ax_birch_freq.set_xlabel("Cluster", color='#7c3aed')
                    ax_birch_freq.set_ylabel("Jumlah Data", color='#7c3aed')
                    ax_birch_freq.grid(axis='y', alpha=0.3, color='#7c3aed')
                    ax_birch_freq.tick_params(colors='#7c3aed')
                    ax_birch_freq.set_facecolor('#ffffff')
                    plt.tight_layout()
                    st.pyplot(fig_birch_freq)
                    
                    birch_freq_df = _build_freq_rows(models['BIRCH'], birch_freq)
                    st.dataframe(birch_freq_df, use_container_width=True, hide_index=True)
                
                cluster_summary_df = pd.DataFrame()
                cluster_detail_df = pd.DataFrame()

                if all(column_name is not None for _, column_name in cluster_attribute_specs):
                    st.subheader(f"Ringkasan Pembelian per Cluster dan Jenis Sepeda dengan {best_model_name}")
                    st.caption(
                        "Tabel ringkasan menampilkan total penjualan per jenis sepeda di dalam cluster, "
                        "sedangkan tabel detail menampilkan lokasi dan distribusi gender pembeli."
                    )
                    # Siapkan quantity_series dari data asli (sebelum standarisasi)
                    _qty_col_for_table = find_matching_column(df_raw_after_currency.columns, ('quantity',)) or find_matching_column(df_raw_after_currency.columns, ('qty',))
                    _qty_series_for_table = df_raw_after_currency[_qty_col_for_table].copy() if _qty_col_for_table is not None else None

                    cluster_summary_df, cluster_detail_df = build_cluster_purchase_tables(
                        df,
                        best_labels,
                        cluster_attribute_specs,
                        price_series=price_display_series,
                        quantity_series=_qty_series_for_table,
                    )
                    summary_display_df = cluster_summary_df.rename(columns={
                        'Bike Model': 'Jenis Sepeda',
                    })
                    st.dataframe(summary_display_df, use_container_width=True, hide_index=True)

                    if not cluster_detail_df.empty:
                        with st.expander("Lihat detail pembelian per cluster", expanded=False):
                            detail_display_df = cluster_detail_df.rename(columns={
                                'Bike Model': 'Jenis Sepeda',
                            })
                            st.dataframe(detail_display_df, use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        "Kolom Bike Model, Store Location, dan Gender tidak ditemukan untuk membuat ringkasan pembelian per cluster."
                    )
                
                # Tampilkan preview hasil clustering - data mentah per baris
                st.subheader(f"Preview Pembelian per Cluster dan Jenis Sepeda dengan {best_model_name}")

                # Identifikasi kolom-kolom yang diperlukan dari data asli
                _price_col_raw   = find_matching_column(df_raw_after_currency.columns, ('price',)) or find_matching_column(df_raw_after_currency.columns, ('harga',))
                _qty_col_raw     = find_matching_column(df_raw_after_currency.columns, ('quantity',)) or find_matching_column(df_raw_after_currency.columns, ('qty',))
                _bike_col_raw    = find_matching_column(df_raw_after_currency.columns, ('bike', 'model'))
                _store_col_raw   = find_matching_column(df_raw_after_currency.columns, ('store', 'location'))
                _gender_col_raw  = find_matching_column(df_raw_after_currency.columns, ('gender',))

                _raw_preview_cols = []
                _raw_col_rename   = {}
                if _price_col_raw:
                    _raw_preview_cols.append(_price_col_raw)
                    _raw_col_rename[_price_col_raw] = 'Price (Harga dalam Rupiah)'
                if _qty_col_raw:
                    _raw_preview_cols.append(_qty_col_raw)
                    _raw_col_rename[_qty_col_raw] = 'Quantity'
                if _bike_col_raw:
                    _raw_preview_cols.append(_bike_col_raw)
                    _raw_col_rename[_bike_col_raw] = 'Bike Model'
                if _store_col_raw:
                    _raw_preview_cols.append(_store_col_raw)
                    _raw_col_rename[_store_col_raw] = 'Store Location'
                if _gender_col_raw:
                    _raw_preview_cols.append(_gender_col_raw)
                    _raw_col_rename[_gender_col_raw] = 'Gender'

                if _raw_preview_cols:
                    # Buat dataframe dari data mentah + label cluster BIRCH
                    birch_labels_raw = models['BIRCH']
                    _raw_preview_df = df_raw_after_currency[_raw_preview_cols].copy()
                    _raw_preview_df = _raw_preview_df.rename(columns=_raw_col_rename)
                    _raw_preview_df['Cluster'] = [f'Cluster {lbl}' for lbl in birch_labels_raw]
                    # Urutkan berdasarkan Cluster
                    _raw_preview_df = _raw_preview_df.sort_values('Cluster', kind='mergesort').reset_index(drop=True)
                    # Format harga sebagai rupiah untuk tampilan preview
                    if 'Price (Harga dalam Rupiah)' in _raw_preview_df.columns:
                        _price_display_col_cfg = st.column_config.NumberColumn(
                            'Price (Harga dalam Rupiah)',
                            format='Rp %.0f',
                        )
                    else:
                        _price_display_col_cfg = None
                    _col_cfg_preview = {}
                    if _price_display_col_cfg is not None:
                        _col_cfg_preview['Price (Harga dalam Rupiah)'] = _price_display_col_cfg
                    st.dataframe(
                        _raw_preview_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=_col_cfg_preview,
                    )
                else:
                    st.info("Tidak ada data preview pembelian yang bisa ditampilkan.")

                # Prepare data for Excel/CSV download
                # Sheet utama: data mentah dengan kolom Price, Quantity, Bike Model, Store Location, Gender, Cluster (BIRCH)
                if _raw_preview_cols:
                    # _raw_preview_df sudah dibuat di atas (data mentah + cluster BIRCH)
                    download_main_df = _raw_preview_df.copy()
                else:
                    # Fallback ke selected features + cluster jika kolom tidak ditemukan
                    download_main_df = df[[*display_features, cluster_column_name]].copy()
                    numeric_cols_export = download_main_df.select_dtypes(include=np.number).columns.tolist()
                    numeric_cols_export = [col for col in numeric_cols_export if col != cluster_column_name]
                    download_main_df[numeric_cols_export] = download_main_df[numeric_cols_export].round(3)

                # export to in-memory Excel file and provide download (avoid writing to disk)
                buffer = BytesIO()
                # choose available Excel engine: prefer xlsxwriter, fallback to openpyxl
                excel_engine = None
                try:
                    import xlsxwriter  # type: ignore
                    excel_engine = 'xlsxwriter'
                except Exception:
                    try:
                        import openpyxl  # type: ignore
                        excel_engine = 'openpyxl'
                    except Exception:
                        excel_engine = None

                if excel_engine is not None:
                    with pd.ExcelWriter(buffer, engine=excel_engine) as writer:
                        download_main_df.to_excel(writer, sheet_name="Data Hasil", index=False)
                        if not cluster_summary_df.empty:
                            cluster_summary_df.to_excel(writer, sheet_name="Ringkasan Pembelian", index=False)
                        if not cluster_detail_df.empty:
                            cluster_detail_df.to_excel(writer, sheet_name="Detail Pembelian", index=False)
                    buffer.seek(0)
                    st.download_button(
                        "Download Hasil dalam Excel",
                        buffer,
                        file_name="hasil_clustering.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
                else:
                    # both engines missing — provide CSV fallback and instruct how to install
                    csv_buf = BytesIO()
                    csv_buf.write(download_main_df.to_csv(index=False).encode('utf-8'))
                    csv_buf.seek(0)
                    st.warning("Module 'xlsxwriter' atau 'openpyxl' tidak ditemukan — menyediakan unduhan CSV sebagai fallback. Untuk Excel install: pip install xlsxwriter atau pip install openpyxl")
                    st.download_button(
                        "Download Hasil dalam CSV",
                        csv_buf,
                        file_name="hasil_clustering.csv",
                        mime='text/csv'
                    )

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>Code By Corneliezmann 2026</div>",
    unsafe_allow_html=True
)
