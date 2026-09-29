from __future__ import annotations

import json
import logging
import time
from html import escape
from datetime import date, datetime, time as clock_time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from youtube_analytics.config import ARTIFACT_DIR, DATA_PATH, LOGGER, MODEL_PATH
from youtube_analytics.data import MODEL_NUMERIC_FEATURES, load_dataset, make_prediction_row, topic_analysis
from youtube_analytics.database import load_prediction_history, persist_dataset, save_model_metrics, save_prediction
from youtube_analytics.modeling import explain_model, load_model_bundle, predict_video, train_models


st.set_page_config(
    page_title="YouTube Trend Intelligence",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "bg": "#08111c", "panel": "#0d1926", "line": "#20344a", "text": "#edf4ff",
    "muted": "#93a4b9", "blue": "#55a8ff", "mint": "#55d6a3", "amber": "#ffb351",
    "pink": "#ee71bc", "purple": "#a88bff", "red": "#ff6475", "teal": "#4dd8cf",
}
CHART_TEMPLATE = "plotly_dark"
NAV_PAGES = [
    "Overview", "Trend Analytics", "ML Predictions", "NLP Analysis", "Channel Analysis",
    "Category Analysis", "Engagement Metrics", "View Forecast", "Recommendation Engine",
    "Model Explainability", "Data Explorer",
]


def inject_theme() -> None:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 78% 0%, #12253a 0%, #08111c 40%, #07101a 100%); color: #edf4ff; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #091420; border-right: 1px solid #1b2d40; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aab9cb; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: 0 !important; }
    h1 { font-size: 1.65rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: .95rem !important; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, #122235, #0d1926); border: 1px solid #233a51; border-radius: 8px; padding: 14px 15px; min-height: 112px; }
    [data-testid="stMetricLabel"] { color: #a5b4c8 !important; font-size: .78rem !important; }
    [data-testid="stMetricValue"] { color: #f4f8ff !important; font-family: 'Manrope',sans-serif; font-size: 1.55rem !important; }
    [data-testid="stMetricDelta"] { font-size: .72rem !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg,#101d2b,#0b1622); border-color: #20344a; border-radius: 8px; }
    .eyebrow { color: #8fa2b9; letter-spacing: 1.4px; font-size: .65rem; font-weight: 700; }
    .subhead { color: #9baec2; font-size: .82rem; margin-top: -.35rem; }
    .status-pill { display:inline-flex; align-items:center; gap:6px; border:1px solid #28563f; background:#123322; color:#62dca7; border-radius:20px; padding:4px 9px; font-size:.68rem; font-weight:700; }
    .status-dot { width:6px; height:6px; background:#62dca7; border-radius:50%; box-shadow:0 0 8px #62dca7; }
    .idea-card { border:1px solid #28415b; background:linear-gradient(120deg,#112539,#101925); border-radius:7px; padding:13px 15px; margin:.4rem 0; }
    .idea-card b { color:#eff6ff; font-size:.85rem; }
    .idea-card p { color:#9eb0c4; font-size:.74rem; margin:.35rem 0 0; }
    .insight-card { border-left:3px solid #ffbc57; background:#211d16; padding:10px 13px; margin:.4rem 0; border-radius:0 6px 6px 0; color:#ead9b3; font-size:.78rem; }
    .small-muted { color:#8c9fb5; font-size:.72rem; }
    .slicer-count { padding:8px 10px; border:1px solid #27415c; border-radius:7px; background:linear-gradient(110deg,#132638,#0c1723); color:#9fb2c9; font-size:.73rem; box-shadow:0 7px 20px #02081155; }
    .filter-summary { display:flex; flex-wrap:wrap; align-items:center; gap:6px; margin:8px 0 1px; }
    .filter-summary-label { margin-right:3px; color:#8298b0; font-size:.62rem; font-weight:700; letter-spacing:1px; }
    .filter-chip { display:inline-flex; gap:5px; align-items:center; border:1px solid #29435d; border-radius:99px; background:linear-gradient(120deg,#15283b,#101c29); padding:4px 9px; color:#c3d1df; font-size:.67rem; box-shadow:0 3px 10px #02081135; }
    .filter-chip b { color:#70bcff; font-weight:700; }
    .filter-chip.filter-unavailable { border-style:dashed; color:#8999ac; background:#101924; }
    .filter-chip.filter-unavailable b { color:#a3afbd; }
    .country-unavailable { min-height:66px; padding:8px 9px; border:1px dashed #32475d; border-radius:6px; color:#a6b7c8; font-size:.78rem; }
    .country-unavailable small { color:#8396ab; font-size:.67rem; }
    [data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMultiSelect"]) { background:linear-gradient(145deg,#132336d9,#0c1725e8); box-shadow:0 9px 28px #02081250; backdrop-filter:blur(14px); }
    .stButton > button { border-radius:6px; border-color:#314b66; }
    .stButton > button[kind="primary"] { border-color:#1887f2; background:#1677df; }
    [data-testid="stDataFrame"] { border:1px solid #20344a; border-radius:7px; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div { border-color:#293e54; background:#0c1825; }
    .block-container { padding-top:1.35rem; padding-bottom:2rem; max-width:1600px; }
    </style>
    """, unsafe_allow_html=True)


def plot_layout(fig: go.Figure, height: int = 330, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template=CHART_TEMPLATE, height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font={"family": "DM Sans", "color": COLORS["muted"], "size": 11},
        margin={"l": 12, "r": 14, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 10}, "bgcolor": "rgba(0,0,0,0)"} if legend else {"visible": False},
        hoverlabel={"bgcolor": "#122235", "bordercolor": "#39536e", "font": {"color": "#edf4ff"}},
    )
    fig.update_xaxes(gridcolor="#1d3044", linecolor="#20344a", zerolinecolor="#20344a", tickfont={"size": 9})
    fig.update_yaxes(gridcolor="#1d3044", linecolor="#20344a", zerolinecolor="#20344a", tickfont={"size": 9})
    return fig


@st.cache_data(show_spinner="Reading and validating INvideos.csv…")
def get_dataset(data_path: str, modified: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    del modified
    return load_dataset(Path(data_path))


@st.cache_resource(show_spinner=False)
def get_saved_bundle(model_path: str, modified: float) -> dict[str, Any] | None:
    del modified
    return load_model_bundle(Path(model_path))


def make_demo_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2021-01-01", periods=800, freq="2D")
    categories = ["Entertainment", "Music", "Gaming", "Technology", "Education", "Sports"]
    frame = pd.DataFrame({
        "video_id": [f"demo_{index}" for index in range(len(dates))],
        "title": [f"{rng.choice(['Amazing', 'Best', 'New', 'How to'])} {rng.choice(['AI tools', 'music mix', 'gaming guide', 'tech review'])} {index}" for index in range(len(dates))],
        "channel_title": [f"Creator {index % 35:02}" for index in range(len(dates))],
        "category_name": rng.choice(categories, len(dates)), "category_id": rng.integers(1, 30, len(dates)),
        "published_at": dates + pd.to_timedelta(rng.integers(0, 24, len(dates)), unit="h"),
        "views": rng.lognormal(11, 1.2, len(dates)).astype(int), "likes": rng.integers(10, 40000, len(dates)),
        "dislikes": rng.integers(0, 1500, len(dates)), "comments": rng.integers(1, 7000, len(dates)),
        "duration_seconds": rng.integers(30, 3600, len(dates)), "tags": "demo, video, analysis",
        "description": "A demonstration video with tips, information and analysis.",
        "thumbnail_score": rng.uniform(25, 95, len(dates)), "trending": rng.binomial(1, .22, len(dates)),
        "trend_days": rng.integers(0, 8, len(dates)),
    })
    from youtube_analytics.data import clean_and_engineer
    return clean_and_engineer(frame)


def cached_frame(frame: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame
    return frame.sample(sample_size, random_state=42)


def compute_insights(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["The current filters have no matching videos. Widen the date or metric range to surface patterns."]
    insights: list[str] = []
    category = frame.groupby("category_name")["trending"].mean().sort_values(ascending=False)
    if not category.empty:
        insights.append(f"{category.index[0]} leads the filtered categories with a {category.iloc[0]:.1%} trend rate.")
    if frame["published_at"].nunique() > 1:
        weekday = frame.groupby("published_weekday")["engagement_rate"].median().idxmax()
        weekday_name = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(weekday))).day_name()
        insights.append(f"{weekday_name} has the strongest median engagement in this selection.")
    trending = frame.loc[frame["trending"].eq(1), "views"]
    nontrending = frame.loc[frame["trending"].eq(0), "views"]
    if not trending.empty and not nontrending.empty:
        lift = max(float(trending.median() / max(nontrending.median(), 1)), 0)
        insights.append(f"Trending videos have {lift:.1f}× the median views of non-trending videos in this selection.")
    peak_hour = int(frame.groupby("published_hour")["trending"].mean().idxmax())
    insights.append(f"Publication hour {peak_hour:02d}:00 has the highest observed trend rate among these videos.")
    return insights[:4]


FILTER_KEYS = [
    "filter_dates", "filter_category", "filter_channel", "filter_country", "filter_video_type",
    "filter_engagement_level", "filter_views", "filter_likes", "filter_comments",
    "filter_engagement", "filter_trending", "filter_prediction_status", "filter_category_cleared",
    "filter_channel_cleared", "filter_country_cleared",
]


def reset_filter_state() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def set_filter_selection(key: str, values: list[str]) -> None:
    st.session_state[key] = values
    st.session_state[f"{key}_cleared"] = not bool(values)


def mark_filter_selection_changed(key: str) -> None:
    st.session_state[f"{key}_cleared"] = not bool(st.session_state.get(key, []))


def engagement_bands(frame: pd.DataFrame) -> pd.Series:
    low_cut, high_cut = frame["engagement_rate"].quantile([1 / 3, 2 / 3])
    return pd.Series(
        np.select(
            [frame["engagement_rate"].le(low_cut), frame["engagement_rate"].le(high_cut)],
            ["Low", "Medium"],
            default="High",
        ),
        index=frame.index,
        dtype="object",
    )


def apply_slicer_filters(frame: pd.DataFrame, selections: dict[str, Any]) -> pd.DataFrame:
    """Apply all shared dashboard slicers to a cleaned source frame."""
    source_engagement_bands = engagement_bands(frame)
    source_durations = frame["duration_seconds"]
    source_video_types = pd.Series(
        np.select([source_durations.le(60), source_durations.le(1200)], ["Short (≤ 60 sec)", "Standard (1–20 min)"], default="Long (> 20 min)"),
        index=frame.index,
    )
    filtered = frame
    date_range = selections.get("dates")
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            filtered = filtered.loc[filtered["published_at"].dt.date.between(start_date, end_date)]

    for key, column in [("categories", "category_name"), ("channels", "channel_title"), ("countries", "country")]:
        selected = selections.get(key)
        if selections.get(f"{key}_cleared", False):
            return filtered.iloc[0:0].copy()
        if selected and column in filtered.columns:
            filtered = filtered.loc[filtered[column].astype(str).isin([str(value) for value in selected])]

    video_types = selections.get("video_types")
    if video_types:
        filtered = filtered.loc[source_video_types.loc[filtered.index].isin(video_types)]

    levels = selections.get("engagement_levels")
    if levels:
        filtered = filtered.loc[source_engagement_bands.loc[filtered.index].isin(levels)]

    for key, column in [("views", "views"), ("likes", "likes"), ("comments", "comments"), ("engagement", "engagement_rate")]:
        value_range = selections.get(key)
        if value_range and len(value_range) == 2:
            filtered = filtered.loc[filtered[column].between(float(value_range[0]), float(value_range[1]))]

    trending = selections.get("trending", "All")
    if trending == "Trending":
        filtered = filtered.loc[filtered["trending"].eq(1)]
    elif trending == "Not trending":
        filtered = filtered.loc[filtered["trending"].eq(0)]

    prediction_status = selections.get("prediction_status", "All")
    if prediction_status in {"Predicted trending", "Predicted not trending"} and "_predicted_trending" in filtered:
        desired = prediction_status == "Predicted trending"
        filtered = filtered.loc[filtered["_predicted_trending"].eq(desired)]
    return filtered.copy()


@st.cache_data(show_spinner="Scoring videos with the saved trend model…", max_entries=4)
def get_prediction_labels(features: pd.DataFrame, model_stamp: float, _bundle: dict[str, Any]) -> pd.DataFrame:
    del model_stamp
    prepared = features.copy()
    prepared["title_text"] = prepared["title"].fillna("")
    matrix = _bundle["preprocessor"].transform(prepared)
    classifier = _bundle["classifiers"].get("XGBoost") or _bundle["classifiers"].get("Random Forest") or next(iter(_bundle["classifiers"].values()))
    probabilities = classifier.predict_proba(matrix)[:, 1]
    holdout_mask = pd.to_datetime(prepared["published_at"]).ge(pd.Timestamp(_bundle["cutoff"]))
    return pd.DataFrame({
        "video_id": prepared["video_id"].to_numpy(),
        "_predicted_trending": np.where(holdout_mask, probabilities >= 0.5, None),
        "_trend_probability": np.where(holdout_mask, probabilities, np.nan),
    })


def filter_summary(selections: dict[str, Any], available_countries: bool, prediction_available: bool) -> str:
    chips = []
    for label, key in [("Category", "categories"), ("Creator", "channels"), ("Country", "countries"), ("Video type", "video_types"), ("Engagement", "engagement_levels")]:
        values = selections.get(key) or []
        if values:
            shown = ", ".join(escape(str(value)) for value in values[:3])
            remainder = f" +{len(values) - 3}" if len(values) > 3 else ""
            chips.append(f"<span class='filter-chip'><b>{label}</b> {shown}{remainder}</span>")
        elif selections.get(f"{key}_cleared", False):
            chips.append(f"<span class='filter-chip'><b>{label}</b> None</span>")
        elif key == "countries" and not available_countries:
            chips.append("<span class='filter-chip filter-unavailable'><b>Country</b> not in source</span>")
        else:
            chips.append(f"<span class='filter-chip'><b>{label}</b> All</span>")
    for label, key in [("Views", "views"), ("Likes", "likes"), ("Comments", "comments"), ("Engagement", "engagement")]:
        value_range = selections.get(key)
        if value_range:
            chips.append(f"<span class='filter-chip'><b>{label}</b> {float(value_range[0]):,.3g}–{float(value_range[1]):,.3g}</span>")
    for label, key in [("Trending", "trending"), ("Prediction", "prediction_status")]:
        value = selections.get(key, "All")
        if value != "All":
            chips.append(f"<span class='filter-chip'><b>{label}</b> {escape(str(value))}</span>")
    if not prediction_available:
        chips.append("<span class='filter-chip filter-unavailable'><b>Prediction</b> train model to enable</span>")
    dates = selections.get("dates")
    if dates and len(dates) == 2:
        chips.insert(0, f"<span class='filter-chip'><b>Published</b> {dates[0]}–{dates[1]}</span>")
    return "".join(chips)


def global_filters(frame: pd.DataFrame, bundle: dict[str, Any] | None = None) -> pd.DataFrame:
    st.markdown("<span class='eyebrow'>INTERACTIVE SLICERS</span>", unsafe_allow_html=True)
    reset_col, count_col = st.columns([1, 5])
    with reset_col:
        st.button("↺ Reset filters", on_click=reset_filter_state, width="stretch", key="reset_filters")
    with count_col:
        st.markdown("<div class='slicer-count'>Filters apply across every page · search options by typing · changes update charts immediately</div>", unsafe_allow_html=True)

    min_date = frame["published_at"].min().date()
    max_date = frame["published_at"].max().date()
    categories = sorted(frame["category_name"].dropna().unique().tolist())
    channels = sorted(frame["channel_title"].dropna().unique().tolist())
    has_country = "country" in frame.columns and frame["country"].notna().any()
    countries = sorted(frame["country"].dropna().astype(str).unique().tolist()) if has_country else []
    with st.container(border=True):
        date_col, category_col, channel_col, country_col = st.columns([1.05, 1.5, 1.6, 1.1])
        with date_col:
            date_range = st.date_input("📅 Upload / published date", value=(min_date, max_date), min_value=min_date, max_value=max_date, key="filter_dates")
        with category_col:
            st.multiselect("📂 Video category", categories, key="filter_category", on_change=mark_filter_selection_changed, args=("filter_category",), placeholder="All categories · type to search")
            select_col, clear_col = st.columns(2)
            select_col.button("Select all", key="select_all_categories", on_click=set_filter_selection, args=("filter_category", categories), width="stretch")
            clear_col.button("Clear all", key="clear_categories", on_click=set_filter_selection, args=("filter_category", []), width="stretch")
        with channel_col:
            st.multiselect("▶️ Channel / creator", channels, key="filter_channel", on_change=mark_filter_selection_changed, args=("filter_channel",), placeholder="All creators · search by name")
            select_col, clear_col = st.columns(2)
            select_col.button("Select all", key="select_all_channels", on_click=set_filter_selection, args=("filter_channel", channels), width="stretch")
            clear_col.button("Clear all", key="clear_channels", on_click=set_filter_selection, args=("filter_channel", []), width="stretch")
        if has_country:
            with country_col:
                st.multiselect("🌍 Country / region", countries, key="filter_country", on_change=mark_filter_selection_changed, args=("filter_country",), placeholder="All regions · type to search")
                select_col, clear_col = st.columns(2)
                select_col.button("Select all", key="select_all_countries", on_click=set_filter_selection, args=("filter_country", countries), width="stretch")
                clear_col.button("Clear all", key="clear_countries", on_click=set_filter_selection, args=("filter_country", []), width="stretch")
        else:
            with country_col:
                st.markdown("<div class='country-unavailable'>🌍 Country / region<br><small>Unavailable in source CSV</small></div>", unsafe_allow_html=True)

        with st.expander("⚙️ Advanced filters · engagement, ranges, trend and model status", expanded=False):
            type_options = ["Short (≤ 60 sec)", "Standard (1–20 min)", "Long (> 20 min)"]
            level_options = ["Low", "Medium", "High"]
            advanced_left, advanced_right = st.columns(2)
            with advanced_left:
                type_col, level_col = st.columns(2)
                type_col.multiselect("🎯 Video type", type_options, key="filter_video_type", placeholder="All video types")
                level_col.multiselect("📈 Engagement level", level_options, key="filter_engagement_level", placeholder="All engagement levels")
                trending_col, prediction_col = st.columns(2)
                trending_col.selectbox("🔥 Trending status", ["All", "Trending", "Not trending"], key="filter_trending")
                prediction_options = ["All", "Predicted trending", "Predicted not trending"]
                prediction_col.selectbox("🤖 Prediction status", prediction_options, key="filter_prediction_status", disabled=bundle is None, help="Train a model on ML Predictions to enable scored-status filtering.")
                if bundle is None:
                    st.caption("Prediction status becomes available after a model suite has been trained.")
            with advanced_right:
                st.markdown("**👁️ View · 👍 Like · 💬 Comment ranges**")
                views_range = numeric_slicer(frame["views"], "Views", "filter_views", integer=True)
                likes_range = numeric_slicer(frame["likes"], "Likes", "filter_likes", integer=True)
                comments_range = numeric_slicer(frame["comments"], "Comments", "filter_comments", integer=True)
                engagement_range = numeric_slicer(frame["engagement_rate"], "Engagement rate", "filter_engagement", format_string="%.3f")

    selections: dict[str, Any] = {
        "dates": date_range if isinstance(date_range, (tuple, list)) and len(date_range) == 2 else (min_date, max_date),
        "categories": st.session_state.get("filter_category", []),
        "channels": st.session_state.get("filter_channel", []),
        "countries": st.session_state.get("filter_country", []) if has_country else [],
        "categories_cleared": st.session_state.get("filter_category_cleared", False),
        "channels_cleared": st.session_state.get("filter_channel_cleared", False),
        "countries_cleared": st.session_state.get("filter_country_cleared", False),
        "video_types": st.session_state.get("filter_video_type", []),
        "engagement_levels": st.session_state.get("filter_engagement_level", []),
        "views": views_range if "views_range" in locals() else full_numeric_range(frame["views"]),
        "likes": likes_range if "likes_range" in locals() else full_numeric_range(frame["likes"]),
        "comments": comments_range if "comments_range" in locals() else full_numeric_range(frame["comments"]),
        "engagement": engagement_range if "engagement_range" in locals() else full_numeric_range(frame["engagement_rate"]),
        "trending": st.session_state.get("filter_trending", "All"),
        "prediction_status": st.session_state.get("filter_prediction_status", "All"),
    }
    filtered = apply_slicer_filters(frame, selections)
    prediction_status = selections["prediction_status"]
    if prediction_status != "All" and bundle is not None and not filtered.empty:
        feature_columns = ["video_id", "title", "category_name", *MODEL_NUMERIC_FEATURES]
        feature_frame = frame[feature_columns].copy()
        model_stamp = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else float(bundle.get("trained_seconds", 0))
        labels = get_prediction_labels(feature_frame, model_stamp, bundle)
        filtered = filtered.merge(labels, on="video_id", how="left", validate="one_to_one")
        desired = prediction_status == "Predicted trending"
        filtered = filtered.loc[filtered["_predicted_trending"].eq(desired)]

    summary_html = filter_summary(selections, has_country, bundle is not None)
    st.markdown(f"<div class='filter-summary'><span class='filter-summary-label'>SELECTED FILTERS</span>{summary_html}</div>", unsafe_allow_html=True)
    st.caption(f"Showing **{len(filtered):,}** of **{len(frame):,}** validated records · filters synchronize across all dashboard pages")
    return filtered


def full_numeric_range(values: pd.Series) -> tuple[float, float]:
    lower, upper = float(values.min()), float(values.max())
    if lower >= upper:
        upper = lower + 1.0
    return lower, upper


def numeric_slicer(
    values: pd.Series,
    label: str,
    key: str,
    *,
    integer: bool = False,
    format_string: str = "%d",
) -> tuple[float, float]:
    lower, upper = full_numeric_range(values)
    if integer:
        lower, upper = int(lower), int(upper)
        step = max(int((upper - lower) / 1000), 1)
        return st.slider(label, min_value=lower, max_value=upper, value=(lower, upper), step=step, key=key)
    return st.slider(label, min_value=lower, max_value=upper, value=(lower, upper), step=(upper - lower) / 1000, format=format_string, key=key)


def page_header(page: str, note: str) -> None:
    left, right = st.columns([4, 1])
    with left:
        st.markdown("<div class='eyebrow'>CREATOR INTELLIGENCE · INDIA DATASET</div>", unsafe_allow_html=True)
        st.title(page)
        st.markdown(f"<div class='subhead'>{note}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div style='text-align:right;padding-top:12px'><span class='status-pill'><i class='status-dot'></i> PIPELINE READY</span></div>", unsafe_allow_html=True)


def metric_row(frame: pd.DataFrame) -> None:
    total_views = frame["views"].sum()
    metrics = st.columns(6)
    values = [
        ("Videos analyzed", f"{len(frame):,}", None),
        ("Total views", f"{total_views / 1e9:.2f}B" if total_views >= 1e9 else f"{total_views / 1e6:.1f}M", None),
        ("Trending videos", f"{int(frame['trending'].sum()):,}", f"{frame['trending'].mean():.1%} of selection" if len(frame) else ""),
        ("Median views", f"{frame['views'].median():,.0f}" if len(frame) else "0", None),
        ("Median engagement", f"{frame['engagement_rate'].median():.1%}" if len(frame) else "0%", None),
        ("Channels", f"{frame['channel_title'].nunique():,}", None),
    ]
    for column, (label, value, delta) in zip(metrics, values):
        column.metric(label, value, delta)


def draw_overview(frame: pd.DataFrame, clean_report: dict[str, Any], geo_available: bool) -> None:
    page_header("YouTube Trend Intelligence", "Analyze historical performance, forecast what may trend, and turn patterns into practical publishing ideas.")
    metric_row(frame)
    st.markdown("")
    left, right = st.columns([1.55, 1])
    with left, st.container(border=True):
        st.subheader("Trending videos over time")
        if frame.empty:
            st.info("No videos match the active filters.")
        else:
            trend = frame.set_index("published_at").resample("MS").agg(videos=("video_id", "count"), views=("views", "sum"), trending=("trending", "sum")).reset_index()
            figure = go.Figure()
            figure.add_trace(go.Scatter(x=trend["published_at"], y=trend["views"], name="Views", mode="lines", line={"color": COLORS["blue"], "width": 2.4}, fill="tozeroy", fillcolor="rgba(85,168,255,.08)"))
            figure.add_trace(go.Scatter(x=trend["published_at"], y=trend["trending"], name="Trending videos", mode="lines", yaxis="y2", line={"color": COLORS["pink"], "width": 2, "shape": "spline"}))
            figure.update_layout(yaxis={"title": "Views", "tickformat": ",.2s"}, yaxis2={"title": "Trending", "overlaying": "y", "side": "right", "showgrid": False}, hovermode="x unified")
            st.plotly_chart(plot_layout(figure, 330), width="stretch", key="overview_trend")
    with right, st.container(border=True):
        st.subheader("Video mix by category")
        if frame.empty:
            st.info("No category data for this selection.")
        else:
            category = frame.groupby("category_name").agg(videos=("video_id", "count"), views=("views", "sum")).reset_index().sort_values("videos", ascending=False)
            figure = px.pie(category, names="category_name", values="videos", hole=.65, color_discrete_sequence=px.colors.qualitative.Bold, custom_data=["views"])
            figure.update_traces(textinfo="percent", hovertemplate="%{label}<br>%{value:,} videos<br>%{percent}<extra></extra>")
            figure.add_annotation(text=f"{len(frame):,}<br><span style='font-size:11px'>videos</span>", x=.5, y=.5, showarrow=False, font={"size": 19, "color": COLORS["text"]})
            st.plotly_chart(plot_layout(figure, 330, False), width="stretch", key="overview_category_mix")
    lower_left, lower_right = st.columns([1.2, 1])
    with lower_left, st.container(border=True):
        st.subheader("Top channels by views")
        channels = frame.groupby("channel_title").agg(views=("views", "sum"), videos=("video_id", "count")).nlargest(10, "views").reset_index()
        if not channels.empty:
            fig = px.bar(channels.sort_values("views"), x="views", y="channel_title", orientation="h", color="views", color_continuous_scale=[[0, "#315c88"], [1, "#59bdff"]], hover_data=["videos"])
            st.plotly_chart(plot_layout(fig, 315, False), width="stretch", key="overview_channels")
    with lower_right, st.container(border=True):
        st.subheader("Automated signals")
        insights = compute_insights(frame)
        for insight in insights:
            st.markdown(f"<div class='insight-card'>✦ &nbsp;{insight}</div>", unsafe_allow_html=True)
        if geo_available:
            st.caption("Geographic analysis is enabled for the rows that include country metadata.")
        else:
            st.caption("Country-level analysis is hidden because this dataset does not include location fields.")
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset": clean_report.get("source_name", "INvideos.csv"),
            "filtered_rows": len(frame),
            "trending_rows": int(frame["trending"].sum()) if not frame.empty else 0,
            "median_views": float(frame["views"].median()) if not frame.empty else 0,
            "median_engagement_rate": float(frame["engagement_rate"].median()) if not frame.empty else 0,
            "insights": insights,
        }
        st.download_button("Download analyst report", json.dumps(report, indent=2), file_name="youtube_analytics_report.json", mime="application/json", width="stretch")
    with st.expander("Data quality report"):
        st.json(clean_report)


def draw_trend_analytics(frame: pd.DataFrame) -> None:
    page_header("Trend Analytics", "Persistence, breakout behavior, category momentum, outliers, and publishing-time patterns.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    first, second = st.columns(2)
    with first, st.container(border=True):
        st.subheader("Trending persistence")
        duration = frame.loc[frame["trending"].eq(1)].groupby("trend_duration_days").size().rename("videos").reset_index()
        if not duration.empty:
            fig = px.bar(duration, x="trend_duration_days", y="videos", color="videos", color_continuous_scale="Tealgrn", labels={"trend_duration_days": "Days in trending", "videos": "Videos"})
            st.plotly_chart(plot_layout(fig, 300, False), width="stretch", key="trend_persistence")
        else:
            st.info("No trending rows in this selection.")
    with second, st.container(border=True):
        st.subheader("Views distribution and outliers")
        fig = px.box(cached_frame(frame, 12000), x="category_name", y="views", color="trending", color_discrete_map={0: "#4d86c2", 1: COLORS["amber"]}, log_y=True, points="outliers")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(plot_layout(fig, 300), width="stretch", key="trend_outliers")
    third, fourth = st.columns(2)
    with third, st.container(border=True):
        st.subheader("Publication hour × weekday")
        heat = frame.pivot_table(index="published_weekday", columns="published_hour", values="trending", aggfunc="mean")
        heat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(heat.index)]
        fig = px.imshow(heat, color_continuous_scale="Turbo", aspect="auto", labels={"x": "Hour of day", "y": "Published weekday", "color": "Trend rate"}, zmin=0, zmax=max(float(heat.max().max()), .01))
        st.plotly_chart(plot_layout(fig, 330, False), width="stretch", key="trend_publish_heatmap")
    with fourth, st.container(border=True):
        st.subheader("Trend funnel")
        high_views = frame["views"].ge(frame["views"].quantile(.75))
        high_engagement = frame["engagement_rate"].ge(frame["engagement_rate"].quantile(.75))
        qualified = high_views & high_engagement
        steps = [len(frame), int(high_views.sum()), int(qualified.sum()), int((qualified & frame["trending"].eq(1)).sum())]
        fig = go.Figure(go.Funnel(y=["All videos", "Top-quartile views", "Top-quartile engagement", "Marked trending"], x=steps, textinfo="value+percent initial", marker={"color": ["#315d89", "#4386bc", "#4eb7bd", COLORS["mint"]]}))
        st.plotly_chart(plot_layout(fig, 330, False), width="stretch", key="trend_funnel")
    left, right = st.columns([1, 1.2])
    with left, st.container(border=True):
        st.subheader("Performance correlation matrix")
        correlation_columns = ["views", "likes", "dislikes", "comments", "duration_seconds", "engagement_rate", "title_length", "trend_days", "thumbnail_score"]
        correlation = frame[correlation_columns].corr(method="spearman")
        fig = px.imshow(correlation, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f", aspect="auto")
        st.plotly_chart(plot_layout(fig, 390, False), width="stretch", key="trend_correlation")
    with right, st.container(border=True):
        st.subheader("3D feature space")
        space = cached_frame(frame, 2500).copy()
        space["engagement_display"] = space["engagement_rate"].clip(upper=max(float(space["engagement_rate"].quantile(.99)), .01))
        space["trend_label"] = space["trending"].map({0: "Not trending", 1: "Trending"})
        fig = px.scatter_3d(space, x="views_log", y="engagement_display", z="duration_seconds", color="trend_label", size="comments", size_max=7, opacity=.75, color_discrete_map={"Not trending": COLORS["blue"], "Trending": COLORS["mint"]}, hover_name="title", hover_data=["category_name", "channel_title", "views"])
        fig.update_layout(scene={"xaxis_title": "Log views", "yaxis_title": "Engagement", "zaxis_title": "Duration (sec)", "bgcolor": "rgba(0,0,0,0)"})
        st.plotly_chart(plot_layout(fig, 390), width="stretch", key="trend_3d")


def draw_model_metrics(bundle: dict[str, Any] | None, frame: pd.DataFrame, training_frame: pd.DataFrame) -> dict[str, Any] | None:
    page_header("ML Predictions", "Compare time-aware holdout metrics and score prospective uploads. Predictions are estimates, not guarantees.")
    model_file_stamp = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
    bundle = bundle or get_saved_bundle(str(MODEL_PATH), model_file_stamp)
    train_col, info_col = st.columns([1, 2])
    with train_col:
        include_xgb = st.checkbox("Train XGBoost models", value=True, help="XGBoost can take longer on this 50,000-row dataset.")
        sample_fraction = st.select_slider("Training data fraction", options=[0.25, 0.5, 0.75, 1.0], value=0.5, format_func=lambda value: f"{int(value * 100)}%")
        train_button = st.button("Train / refresh model suite", type="primary", width="stretch")
    with info_col:
        if bundle:
            st.info(f"Latest model suite uses a chronological 80/20 split. Holdout starts {bundle['cutoff'][:10]}; {bundle['train_rows']:,} train rows and {bundle['test_rows']:,} holdout rows. Model estimates use title, category, duration, upload time, channel history and historical category priors, not future views/likes/comments.")
        else:
            st.warning("No trained model is saved yet. Train models to enable upload-metadata predictions.")
    if train_button:
        if len(training_frame) < 250:
            st.error("At least 250 source rows are needed for a meaningful time-aware model split.")
        else:
            train_data = training_frame.sort_values("published_at").copy()
            if sample_fraction < 1:
                train_data = train_data.iloc[-max(int(len(train_data) * sample_fraction), 250):]
            try:
                with st.spinner("Training chronological classification and regression models…"):
                    new_bundle, metrics = train_models(train_data, include_xgboost=include_xgb)
                    save_model_metrics(metrics)
                st.cache_resource.clear()
                st.session_state["trained_bundle"] = new_bundle
                st.success(f"Training complete in {new_bundle['trained_seconds']:.1f} seconds. Artifact saved to {MODEL_PATH.name}.")
                bundle = new_bundle
            except Exception as error:
                LOGGER.exception("Model training failed")
                st.error(f"Training could not complete: {error}")
    if bundle is None:
        return None
    metrics = bundle["metrics"].copy()
    selected_task = st.selectbox("Compare task", metrics["task"].dropna().unique().tolist(), key="metric_task")
    task_metrics = metrics.loc[metrics["task"].eq(selected_task)].copy()
    if selected_task == "Trend classification":
        st.caption("Chronological holdout classification performance. Compare recall and precision alongside ROC-AUC for the imbalanced trending target.")
        plot_data = task_metrics.melt(id_vars="model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"], var_name="metric", value_name="score")
        fig = px.bar(plot_data, x="model", y="score", color="metric", barmode="group", color_discrete_sequence=[COLORS["blue"], COLORS["mint"], COLORS["amber"], COLORS["pink"], COLORS["purple"]], range_y=[0, 1])
    else:
        st.caption("Regression metrics measured on the chronological holdout. MAE/RMSE share the target units; R² summarizes explained variance.")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for metric_name, color in [("mae", COLORS["blue"]), ("rmse", COLORS["amber"]), ("r2", COLORS["mint"])]:
            fig.add_trace(go.Bar(x=task_metrics["model"], y=task_metrics[metric_name], name=metric_name.upper(), marker_color=color), secondary_y=metric_name == "r2")
    st.plotly_chart(plot_layout(fig, 350), width="stretch", key="model_metric_comparison")
    st.dataframe(metrics, width="stretch", hide_index=True, column_config={"accuracy": st.column_config.NumberColumn(format="%.3f"), "precision": st.column_config.NumberColumn(format="%.3f"), "recall": st.column_config.NumberColumn(format="%.3f"), "f1": st.column_config.NumberColumn(format="%.3f"), "roc_auc": st.column_config.NumberColumn(format="%.3f"), "mae": st.column_config.NumberColumn(format="%.2f"), "rmse": st.column_config.NumberColumn(format="%.2f"), "r2": st.column_config.NumberColumn(format="%.3f")})
    return bundle


def draw_nlp(frame: pd.DataFrame) -> None:
    page_header("NLP Analysis", "Title and description sentiment, TF-IDF keywords, NMF topic discovery, and content-length patterns.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    sample_limit = st.slider("Text analysis sample", min_value=500, max_value=10000, value=min(3000, max(500, len(frame))), step=500, key="nlp_sample")
    text_frame = frame.sample(min(sample_limit, len(frame)), random_state=42).copy()
    terms = (text_frame["title"] + " " + text_frame["description"].fillna("") + " " + text_frame["tags"].fillna(""))
    with st.spinner("Extracting keywords and topics…"):
        try:
            keywords, topics = topic_analysis(terms, topics=6)
        except ValueError as error:
            st.warning(f"Text analysis needs more distinct terms: {error}")
            return
    first, second = st.columns(2)
    with first, st.container(border=True):
        st.subheader("Top TF-IDF keywords")
        fig = px.treemap(keywords.head(24), path=[px.Constant("Keywords"), "term"], values="weight", color="weight", color_continuous_scale="Tealgrn", hover_data={"weight": ":.3f"})
        st.plotly_chart(plot_layout(fig, 365, False), width="stretch", key="nlp_treemap")
    with second, st.container(border=True):
        st.subheader("NMF topic terms")
        fig = px.bar(topics, x="weight", y="term", color="topic", orientation="h", facet_col="topic", facet_col_wrap=2, color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_yaxes(matches=None, autorange="reversed")
        fig.update_layout(height=365, margin={"l": 5, "r": 5, "t": 40, "b": 10})
        st.plotly_chart(plot_layout(fig, 365), width="stretch", key="nlp_topics")
    sentiment = frame.groupby("trending")["sentiment_score"].median().rename(index={0: "Not trending", 1: "Trending"}).reset_index()
    lengths = frame.assign(length_bin=pd.cut(frame["title_length"], [0, 10, 20, 30, 40, 50, 80, 150], right=True).astype(str)).groupby(["length_bin", "trending"], observed=True).size().rename("videos").reset_index()
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Title sentiment")
        fig = px.violin(cached_frame(frame, 10000), x="trending", y="sentiment_score", color="trending", box=True, points=False, color_discrete_map={0: COLORS["blue"], 1: COLORS["mint"]}, labels={"trending": "Trending status", "sentiment_score": "Lexicon sentiment"})
        st.plotly_chart(plot_layout(fig, 300, False), width="stretch", key="nlp_sentiment")
    with right, st.container(border=True):
        st.subheader("Title length distribution")
        fig = px.bar(lengths, x="length_bin", y="videos", color="trending", barmode="group", color_discrete_map={0: COLORS["blue"], 1: COLORS["amber"]}, labels={"length_bin": "Characters", "videos": "Videos"})
        st.plotly_chart(plot_layout(fig, 300), width="stretch", key="nlp_title_length")
    st.caption("Sentiment is a transparent lightweight lexicon baseline, not a pretrained language model. TF-IDF/NMF topics are descriptive, not causal.")


def draw_channel_analysis(frame: pd.DataFrame) -> None:
    page_header("Channel Analysis", "Compare publishing cadence, historical trend rates, audience reach, and channel-level consistency.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    channel = frame.groupby("channel_title").agg(videos=("video_id", "count"), views=("views", "sum"), median_views=("views", "median"), trend_rate=("trending", "mean"), engagement=("engagement_rate", "median"), active_days=("published_at", "nunique")).reset_index()
    channel = channel.loc[channel["videos"].ge(2)]
    left, right = st.columns([1.2, 1])
    with left, st.container(border=True):
        st.subheader("Channel reach × trend rate")
        fig = px.scatter(channel, x="views", y="trend_rate", size="videos", color="engagement", hover_name="channel_title", log_x=True, color_continuous_scale="Turbo", hover_data={"videos": True, "median_views": ":,.0f", "engagement": ":.2%"})
        st.plotly_chart(plot_layout(fig, 390, False), width="stretch", key="channel_scatter")
    with right, st.container(border=True):
        st.subheader("Top channels by median views")
        top = channel.nlargest(15, "median_views").sort_values("median_views")
        fig = px.bar(top, x="median_views", y="channel_title", orientation="h", color="trend_rate", color_continuous_scale="Mint", hover_data=["videos", "engagement"])
        st.plotly_chart(plot_layout(fig, 390, False), width="stretch", key="channel_bar")
    st.dataframe(channel.sort_values("views", ascending=False), width="stretch", hide_index=True, column_config={"trend_rate": st.column_config.NumberColumn(format="%.1%"), "engagement": st.column_config.NumberColumn(format="%.2%"), "views": st.column_config.NumberColumn(format="%,d"), "median_views": st.column_config.NumberColumn(format="%,.0f")})


def draw_category_analysis(frame: pd.DataFrame) -> None:
    page_header("Category Analysis", "Measure category-level volume, view distribution, trend rate, engagement, and feature balance.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    category = frame.groupby("category_name").agg(videos=("video_id", "count"), views=("views", "sum"), median_views=("views", "median"), trend_rate=("trending", "mean"), engagement=("engagement_rate", "median"), likes=("likes", "median"), comments=("comments", "median"), duration=("trend_duration_days", "median")).reset_index()
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Volume × trend rate")
        fig = px.scatter(category, x="videos", y="trend_rate", size="views", color="engagement", hover_name="category_name", color_continuous_scale="Turbo", size_max=46, hover_data={"median_views": ":,.0f", "engagement": ":.2%"})
        st.plotly_chart(plot_layout(fig, 350), width="stretch", key="category_scatter")
    with right, st.container(border=True):
        st.subheader("Views by category")
        fig = px.treemap(category, path=["category_name"], values="views", color="trend_rate", color_continuous_scale="Tealgrn", hover_data={"videos": True, "trend_rate": ":.1%"})
        st.plotly_chart(plot_layout(fig, 350, False), width="stretch", key="category_treemap")
    radar_categories = category.nlargest(6, "videos").copy()
    normalized = ["median_views", "likes", "comments", "engagement", "trend_rate", "duration"]
    for column in normalized:
        maximum = max(float(radar_categories[column].max()), 1e-9)
        radar_categories[f"{column}_norm"] = radar_categories[column] / maximum * 100
    radar = go.Figure()
    for _, row in radar_categories.iterrows():
        radar.add_trace(go.Scatterpolar(r=[row[f"{col}_norm"] for col in normalized], theta=["Views", "Likes", "Comments", "Engagement", "Trend rate", "Trend duration"], name=row["category_name"], fill="toself", opacity=.4))
    st.plotly_chart(plot_layout(radar, 420), width="stretch", key="category_radar")
    st.dataframe(category.sort_values("views", ascending=False), width="stretch", hide_index=True)


def draw_engagement(frame: pd.DataFrame) -> None:
    page_header("Engagement Metrics", "Understand like, comment, and combined engagement behavior across view tiers and content categories.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    q1, q2, q3, q4 = frame["views"].quantile([.25, .5, .75, .95])
    tiers = frame.assign(view_tier=pd.cut(frame["views"], [-1, q1, q2, q3, q4, np.inf], labels=["Bottom 25%", "25–50%", "50–75%", "75–95%", "Top 5%"], duplicates="drop"))
    agg = tiers.groupby("view_tier", observed=True).agg(videos=("video_id", "count"), engagement=("engagement_rate", "median"), like_rate=("like_rate", "median"), comment_rate=("comment_rate", "median"), views=("views", "median")).reset_index()
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Views and engagement rate by category")
        cat = frame.groupby("category_name").agg(views=("views", "median"), engagement=("engagement_rate", "median")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=cat["category_name"], y=cat["views"], name="Median views", marker_color=COLORS["blue"]), secondary_y=False)
        fig.add_trace(go.Scatter(x=cat["category_name"], y=cat["engagement"], name="Median engagement", mode="lines+markers", line={"color": COLORS["amber"], "width": 2}, marker={"size": 7}), secondary_y=True)
        fig.update_yaxes(title_text="Views", tickformat=",.2s", secondary_y=False)
        fig.update_yaxes(title_text="Engagement", tickformat=".1%", secondary_y=True)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(plot_layout(fig, 390), width="stretch", key="engagement_combo")
    with right, st.container(border=True):
        st.subheader("Engagement across view tiers")
        fig = px.bar(agg, x="view_tier", y=["engagement", "like_rate", "comment_rate"], barmode="group", color_discrete_sequence=[COLORS["mint"], COLORS["blue"], COLORS["pink"]])
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(plot_layout(fig, 390), width="stretch", key="engagement_tiers")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Views × engagement")
        fig = px.scatter(cached_frame(frame, 7000), x="views", y="engagement_rate", color="trending", size="comments", log_x=True, color_discrete_map={0: COLORS["blue"], 1: COLORS["amber"]}, hover_data=["title", "category_name", "channel_title"], labels={"trending": "Trending"})
        st.plotly_chart(plot_layout(fig, 350), width="stretch", key="engagement_scatter")
    with right, st.container(border=True):
        st.subheader("Engagement rate distribution")
        capped = frame.assign(engagement_display=frame["engagement_rate"].clip(upper=frame["engagement_rate"].quantile(.99)))
        fig = px.histogram(capped, x="engagement_display", color="trending", nbins=45, barmode="overlay", opacity=.72, color_discrete_map={0: COLORS["blue"], 1: COLORS["mint"]})
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(plot_layout(fig, 350), width="stretch", key="engagement_histogram")


def draw_view_forecast(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    page_header("View Forecast", "Build a baseline daily forecast from filtered history and compare upload estimates from trained models.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    daily = frame.set_index("published_at").resample("D").agg(views=("views", "sum"), videos=("video_id", "count")).reset_index()
    horizon = st.slider("Forecast horizon (days)", 7, 60, 21, key="forecast_horizon")
    if len(daily) < 14:
        st.info("At least two weeks of non-empty daily history are needed for the forecast view.")
        return
    daily["day_index"] = np.arange(len(daily))
    daily["weekday"] = daily["published_at"].dt.dayofweek
    daily["month"] = daily["published_at"].dt.month
    train = daily.iloc[:-max(7, min(int(len(daily) * .15), 30))]
    holdout = daily.iloc[len(train):]
    design_train = pd.get_dummies(train[["day_index", "weekday", "month"]], columns=["weekday", "month"], dtype=float)
    design_holdout = pd.get_dummies(holdout[["day_index", "weekday", "month"]], columns=["weekday", "month"], dtype=float).reindex(columns=design_train.columns, fill_value=0)
    from sklearn.ensemble import RandomForestRegressor
    forecast_model = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=42, n_jobs=-1)
    forecast_model.fit(design_train, np.log1p(train["views"]))
    holdout["forecast"] = np.maximum(np.expm1(forecast_model.predict(design_holdout)), 0)
    future_dates = pd.date_range(daily["published_at"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future_rows = pd.DataFrame({"published_at": future_dates, "day_index": np.arange(len(daily), len(daily) + horizon)})
    future_rows["weekday"] = future_rows["published_at"].dt.dayofweek
    future_rows["month"] = future_rows["published_at"].dt.month
    future_x = pd.get_dummies(future_rows[["day_index", "weekday", "month"]], columns=["weekday", "month"], dtype=float).reindex(columns=design_train.columns, fill_value=0)
    future_rows["forecast"] = np.maximum(np.expm1(forecast_model.predict(future_x)), 0)
    residual = holdout["views"].to_numpy() - holdout["forecast"].to_numpy()
    deviation = max(float(np.std(residual)), float(np.std(train["views"])), 1.0)
    future_rows["lower"] = np.maximum(future_rows["forecast"] - 1.64 * deviation, 0)
    future_rows["upper"] = future_rows["forecast"] + 1.64 * deviation
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_rows["published_at"], y=future_rows["upper"], line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future_rows["published_at"], y=future_rows["lower"], fill="tonexty", fillcolor="rgba(85,168,255,.15)", line={"width": 0}, name="Approx. 90% band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=daily["published_at"], y=daily["views"], name="Historical views", line={"color": COLORS["blue"], "width": 1.4}))
    fig.add_trace(go.Scatter(x=holdout["published_at"], y=holdout["forecast"], name="Holdout estimate", line={"color": COLORS["amber"], "dash": "dot", "width": 2}))
    fig.add_trace(go.Scatter(x=future_rows["published_at"], y=future_rows["forecast"], name="Forecast estimate", line={"color": COLORS["mint"], "width": 2.5}))
    st.plotly_chart(plot_layout(fig, 430), width="stretch", key="view_forecast_chart")
    holdout_rmse = float(np.sqrt(np.mean(np.square(residual))))
    a, b, c = st.columns(3)
    a.metric("Holdout daily RMSE", f"{holdout_rmse:,.0f} views")
    b.metric("Expected views next 7 days", f"{future_rows.head(7)['forecast'].sum():,.0f}")
    c.metric("Forecast method", "Random forest + calendar")
    st.caption("This aggregate forecast is a historical baseline, not a guarantee. Sparse date ranges, seasonality changes, and platform events can materially affect future views.")
    if bundle:
        st.caption(f"Per-video ML upload estimates are available on ML Predictions. Saved model holdout cutoff: {bundle['cutoff'][:10]}.")


def draw_recommendations(frame: pd.DataFrame) -> None:
    page_header("Recommendation Engine", "Translate filtered historical patterns into category, title, engagement, and publishing-time ideas.")
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    category = frame.groupby("category_name").agg(trend_rate=("trending", "mean"), videos=("video_id", "count"), views=("views", "median"), engagement=("engagement_rate", "median")).reset_index()
    reliable = category.loc[category["videos"].ge(20)].sort_values(["trend_rate", "engagement"], ascending=False)
    best_category = reliable.iloc[0] if not reliable.empty else category.sort_values("trend_rate", ascending=False).iloc[0]
    title_perf = frame.assign(title_band=pd.cut(frame["title_length"], [0, 10, 20, 30, 40, 50, 70, 120], include_lowest=True)).groupby("title_band", observed=True).agg(trend_rate=("trending", "mean"), sample=("video_id", "count")).query("sample >= 15")
    title_band = title_perf["trend_rate"].idxmax() if not title_perf.empty else None
    hour_perf = frame.groupby("published_hour").agg(trend_rate=("trending", "mean"), sample=("video_id", "count"), engagement=("engagement_rate", "median")).query("sample >= 10")
    best_hour = int(hour_perf["trend_rate"].idxmax()) if not hour_perf.empty else int(frame["published_hour"].mode().iloc[0])
    best_weekday = int(frame.groupby("published_weekday")["trending"].mean().idxmax())
    day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][best_weekday]
    c1, c2, c3 = st.columns(3)
    c1.metric("Strongest category", str(best_category["category_name"]), f"{best_category['trend_rate']:.1%} observed trend rate")
    c2.metric("Best observed window", f"{day_name}, {best_hour:02d}:00", "Filtered publication history")
    c3.metric("Higher-signal title length", f"{title_band.left:.0f}–{title_band.right:.0f} chars" if title_band is not None else "More examples needed", "Historical association")
    st.markdown("### Suggested experiments")
    recommendations = [
        ("Choose a category", f"Test a {best_category['category_name']} concept. Its observed trend rate is {best_category['trend_rate']:.1%} across {int(best_category['videos']):,} videos in this filtered history. This association is not a causal lift."),
        ("Shape the title", f"Compare concise, descriptive title variants around {title_band.left:.0f}–{title_band.right:.0f} characters." if title_band is not None else "Collect more examples before recommending a title length."),
        ("Test publishing time", f"A/B test {day_name} near {best_hour:02d}:00, then compare your channel’s results over multiple uploads."),
        ("Tune the content mix", f"The selected videos show a median engagement rate of {best_category['engagement']:.2%} in {best_category['category_name']}. Build a format that invites substantive comments, not just clicks."),
    ]
    for title, body in recommendations:
        st.markdown(f"<div class='idea-card'><b>✦ &nbsp;{title}</b><p>{body}</p></div>", unsafe_allow_html=True)
    st.markdown("### Publishing opportunity heatmap")
    heat = frame.pivot_table(index="published_weekday", columns="published_hour", values="trending", aggfunc="mean")
    fig = px.imshow(heat, color_continuous_scale="Turbo", aspect="auto", labels={"x": "Hour", "y": "Weekday (0 = Monday)", "color": "Trend rate"})
    st.plotly_chart(plot_layout(fig, 300, False), width="stretch", key="recommendation_heatmap")
    st.caption("Recommendations are observational patterns in the selected dataset. Validate them with controlled experiments on your own channel.")


def draw_explainability(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    page_header("Model Explainability", "Explore global feature importance and SHAP explanations for the trained trending classifier.")
    if bundle is None:
        st.warning("Train a model suite first from the ML Predictions page.")
        return
    if frame.empty:
        st.warning("No data under the current filters.")
        return
    sample = cached_frame(frame, 80).copy()
    sample["title_text"] = sample["title"]
    try:
        with st.spinner("Computing feature contributions for a bounded sample…"):
            importance = explain_model(bundle, sample, maximum_rows=80)
    except Exception as error:
        LOGGER.exception("Feature explanation failed")
        st.error(f"Could not compute the explanation: {error}")
        return
    if importance.empty:
        st.warning("The selected classifier does not expose feature importances.")
        return
    st.info("Importance describes model reliance in the current sample; SHAP contributions explain model output, not causality. TF-IDF dimensions are grouped by preprocessing feature names.")
    left, right = st.columns([1.1, .9])
    with left, st.container(border=True):
        st.subheader("Global feature importance")
        fig = px.bar(importance.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=[[0, "#54368f"], [.55, "#557cf2"], [1, "#ff7a78"]])
        st.plotly_chart(plot_layout(fig, 520, False), width="stretch", key="explain_global")
    with right, st.container(border=True):
        st.subheader("Importance share")
        importance["importance_share"] = importance["importance"] / max(importance["importance"].sum(), 1e-12)
        fig = px.treemap(importance.head(20), path=["feature"], values="importance", color="importance_share", color_continuous_scale="Plasma")
        st.plotly_chart(plot_layout(fig, 520, False), width="stretch", key="explain_treemap")
    st.dataframe(importance, width="stretch", hide_index=True)


def draw_data_explorer(frame: pd.DataFrame, clean_report: dict[str, Any]) -> None:
    page_header("Data Explorer", "Inspect, sort, search, export, and review quality signals from the cleaned dataset.")
    search = st.text_input("Search title, channel, description, or category", placeholder="e.g. music, channel_01, tutorial", key="explorer_search")
    explore = frame
    if search:
        mask = pd.Series(False, index=frame.index)
        for column in ["title", "channel_title", "description", "category_name", "tags"]:
            mask |= frame[column].astype(str).str.contains(search, case=False, na=False, regex=False)
        explore = frame.loc[mask]
    sort_column = st.selectbox("Sort rows by", ["published_at", "views", "likes", "comments", "engagement_rate", "trend_days", "title_length"], key="explorer_sort")
    descending = st.toggle("Descending", value=True, key="explorer_desc")
    visible = [column for column in ["video_id", "title", "channel_title", "category_name", "published_at", "views", "likes", "comments", "engagement_rate", "trending", "trend_days", "title_length", "sentiment_score"] if column in explore]
    sorted_rows = explore[visible].sort_values(sort_column, ascending=not descending)
    rows = sorted_rows.head(10000)
    st.caption(f"{len(explore):,} records match current filters/search. The interactive table shows up to 10,000 rows; the CSV download includes all filtered matches.")
    st.dataframe(rows, width="stretch", hide_index=True, height=540, column_config={"engagement_rate": st.column_config.NumberColumn(format="%.2%"), "sentiment_score": st.column_config.NumberColumn(format="%.4f"), "views": st.column_config.NumberColumn(format="%,d"), "likes": st.column_config.NumberColumn(format="%,d"), "comments": st.column_config.NumberColumn(format="%,d"), "published_at": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})
    csv = sorted_rows.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered CSV", csv, file_name="youtube_filtered_data.csv", mime="text/csv", type="primary")
    st.markdown("### Quality and anomalies")
    q1, q2, q3 = st.columns(3)
    q1.metric("Duplicate video IDs removed", f"{clean_report.get('duplicate_rows_removed', 0):,}")
    q2.metric("Invalid dates removed", f"{clean_report.get('invalid_dates_removed', 0):,}")
    q3.metric("Missing cells after cleanup", f"{clean_report.get('missing_cells_after_cleaning', 0):,}")
    anomaly_columns = st.multiselect("Anomaly signals", ["views", "likes", "comments", "duration_seconds", "engagement_rate"], default=["views", "likes", "comments"])
    if anomaly_columns and not explore.empty:
        anomalies = pd.Series(False, index=explore.index)
        for column in anomaly_columns:
            values = explore[column]
            lower, upper = values.quantile([.01, .99])
            anomalies |= values.lt(lower) | values.gt(upper)
        alert_rows = explore.loc[anomalies].sort_values("views", ascending=False).head(100)
        st.caption(f"Robust review queue: {int(anomalies.sum()):,} rows fall outside the selected 1st–99th percentile bands. Extreme does not necessarily mean erroneous.")
        st.dataframe(alert_rows[["video_id", "title", "category_name", *anomaly_columns]].head(50), width="stretch", hide_index=True)


def draw_geo(frame: pd.DataFrame) -> None:
    if "country" not in frame.columns or frame["country"].dropna().empty:
        st.info("Geographic audience maps are not available: INvideos.csv has no country or location column. Add a country field to the source dataset to enable this page.")
        return
    geo = frame.groupby("country").agg(videos=("video_id", "count"), views=("views", "sum"), trend_rate=("trending", "mean")).reset_index()
    fig = px.choropleth(geo, locations="country", locationmode="country names", color="views", hover_name="country", hover_data=["videos", "trend_rate"], color_continuous_scale="Turbo")
    st.plotly_chart(plot_layout(fig, 520), width="stretch", key="geographic_map")


def prediction_form(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    st.markdown("### Score an upload concept")
    if bundle is None:
        st.warning("Train the model suite on the ML Predictions page before scoring an upload.")
        return
    categories = sorted(frame["category_name"].dropna().unique().tolist()) or ["Unknown"]
    channels = ["New channel"] + sorted(frame["channel_title"].dropna().unique().tolist())
    with st.form("video_prediction_form"):
        title = st.text_input("Video title", max_chars=200, placeholder="A clear working title for your video")
        description = st.text_area("Description", max_chars=2500, height=100, placeholder="Draft description or key phrases")
        tags = st.text_input("Tags (comma-separated)", placeholder="AI tools, tutorial, review")
        first, second, third = st.columns(3)
        category = first.selectbox("Category", categories)
        channel = second.selectbox("Channel history", channels)
        duration_minutes = third.number_input("Duration (minutes)", min_value=0.5, max_value=600.0, value=8.0, step=0.5)
        col1, col2, col3 = st.columns(3)
        publish_date = col1.date_input("Planned publish date", value=date.today())
        publish_time = col2.time_input("Planned publish time", value=clock_time(18, 0))
        thumbnail_score = col3.slider("Thumbnail quality estimate", 0, 100, 65)
        submitted = st.form_submit_button("Estimate video performance", type="primary", width="stretch")
    if not submitted:
        return
    if not title.strip():
        st.error("Add a title before requesting an estimate.")
        return
    scheduled = pd.Timestamp(datetime.combine(publish_date, publish_time))
    row = make_prediction_row(title.strip(), description, category, scheduled, duration_minutes * 60, thumbnail_score, frame, channel_title=channel, tags=tags)
    try:
        prediction = predict_video(bundle, row)
        first, second, third, fourth = st.columns(4)
        first.metric("Trend probability", f"{prediction['trend_probability']:.1%}", "Estimated from historical patterns")
        second.metric("Expected views", f"{prediction['expected_views']:,.0f}", "Point estimate")
        third.metric("Engagement rate", f"{prediction['engagement_rate']:.2%}", "Estimated")
        fourth.metric("Trend duration", f"{prediction['trend_duration_days']:.1f} days", "Estimated")
        calibration = bundle["metrics"].loc[bundle["metrics"]["task"].eq("Trend classification"), "roc_auc"]
        st.caption(f"Model trained through {bundle['cutoff'][:10]}. Estimates are not guarantees; holdout ROC-AUC: {calibration.max():.3f}. Aggregate performance reflects a time-based holdout.")
        record = {"created_at": datetime.now().isoformat(timespec="seconds"), "title": title.strip(), "category": category, "channel": channel, **prediction}
        save_prediction(record)
        st.session_state["last_prediction"] = {**record, "schedule": scheduled.isoformat()}
    except Exception as error:
        LOGGER.exception("Video scoring failed")
        st.error(f"Could not generate prediction: {error}")


def draw_model_prediction(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    st.markdown("### Model score for your next upload")
    prediction_form(frame, bundle)
    if st.session_state.get("last_prediction"):
        row = st.session_state["last_prediction"]
        with st.expander(f"Last prediction: {row['title']}", expanded=False):
            st.json(row)
    history = load_prediction_history()
    if not history.empty:
        st.markdown("### Prediction history")
        st.dataframe(history.head(100), width="stretch", hide_index=True)
        st.download_button("Download prediction history", history.to_csv(index=False).encode("utf-8"), file_name="youtube_prediction_history.csv", mime="text/csv")


def draw_filtered_model_results(frame: pd.DataFrame, bundle: dict[str, Any] | None) -> None:
    st.markdown("### Model results for the active slicers")
    if bundle is None:
        st.info("Train a model suite to score the filtered records.")
        return
    if frame.empty:
        st.info("No records match the active slicers.")
        return
    feature_columns = ["video_id", "title", "category_name", "published_at", *MODEL_NUMERIC_FEATURES]
    features = frame[feature_columns].drop_duplicates("video_id")
    model_stamp = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else float(bundle.get("trained_seconds", 0))
    results = get_prediction_labels(features, model_stamp, bundle)
    scored = frame.merge(results, on="video_id", how="inner", validate="one_to_one")
    scored = scored.loc[scored["_trend_probability"].notna()].copy()
    if scored.empty:
        st.info(f"The active filters contain no chronological holdout records. The model holdout begins {bundle['cutoff'][:10]}; training-period rows are intentionally not shown as model predictions.")
        return
    predicted_count = int(scored["_predicted_trending"].fillna(False).astype(bool).sum())
    a, b, c = st.columns(3)
    a.metric("Holdout records scored", f"{len(scored):,}")
    b.metric("Predicted trending", f"{predicted_count:,}", f"{predicted_count / len(scored):.1%} of filtered holdout")
    c.metric("Median trend probability", f"{scored['_trend_probability'].median():.1%}")
    left, right = st.columns([1, 1.35])
    with left, st.container(border=True):
        st.subheader("Predicted trend probability")
        probability_data = scored.assign(prediction=pd.Series(np.where(scored["_predicted_trending"].astype(bool), "Predicted trending", "Predicted not trending"), index=scored.index))
        fig = px.histogram(probability_data, x="_trend_probability", color="prediction", nbins=28, barmode="overlay", opacity=.78, color_discrete_map={"Predicted trending": COLORS["mint"], "Predicted not trending": COLORS["blue"]}, labels={"_trend_probability": "Trend probability"})
        fig.add_vline(x=.5, line_dash="dash", line_color=COLORS["amber"], annotation_text="0.50 threshold")
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(plot_layout(fig, 315), width="stretch", key="filtered_prediction_probability")
    with right, st.container(border=True):
        st.subheader("Highest-scored videos in this selection")
        columns = ["title", "category_name", "channel_title", "_trend_probability", "views", "trending"]
        display = scored.sort_values("_trend_probability", ascending=False)[columns].head(25).rename(columns={"title": "Video title", "category_name": "Category", "channel_title": "Channel", "_trend_probability": "Trend probability", "views": "Observed views", "trending": "Observed trending"})
        st.dataframe(display, width="stretch", hide_index=True, column_config={"Trend probability": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1%"), "Observed views": st.column_config.NumberColumn(format="%,d")})
    st.caption(f"Only chronological holdout rows published on/after {bundle['cutoff'][:10]} are scored here; training rows are excluded to avoid showing in-sample model fit as predictive performance.")


def draw_page(page: str, frame: pd.DataFrame, full_frame: pd.DataFrame, clean_report: dict[str, Any], bundle: dict[str, Any] | None, geo_available: bool) -> None:
    if page == "Overview":
        draw_overview(frame, clean_report, geo_available)
    elif page == "Trend Analytics":
        draw_trend_analytics(frame)
    elif page == "ML Predictions":
        bundle = draw_model_metrics(bundle, frame, full_frame)
        draw_filtered_model_results(frame, bundle)
        st.divider()
        draw_model_prediction(frame, bundle)
    elif page == "NLP Analysis":
        draw_nlp(frame)
    elif page == "Channel Analysis":
        draw_channel_analysis(frame)
    elif page == "Category Analysis":
        draw_category_analysis(frame)
    elif page == "Engagement Metrics":
        draw_engagement(frame)
    elif page == "View Forecast":
        draw_view_forecast(frame, bundle)
    elif page == "Recommendation Engine":
        draw_recommendations(frame)
        st.divider()
        prediction_form(frame, bundle)
    elif page == "Model Explainability":
        draw_explainability(frame, bundle)
    elif page == "Data Explorer":
        draw_data_explorer(frame, clean_report)
        st.divider()
        draw_geo(frame)


def sync_page_query() -> None:
    st.query_params["page"] = st.session_state["nav_page"]


def main() -> None:
    inject_theme()
    st.sidebar.markdown("<div style='display:flex;align-items:center;gap:8px;padding:4px 0 13px'><a href='http://localhost:5173/' aria-label='Go to dashboard home' title='Home' style='width:30px;height:30px;display:grid;place-items:center;flex:0 0 auto;border:1px solid #29445f;border-radius:7px;background:#102033;color:#9dccff;text-decoration:none;transition:background .18s,border-color .18s' onmouseover=\"this.style.background='#18324b';this.style.borderColor='#4b91c7'\" onmouseout=\"this.style.background='#102033';this.style.borderColor='#29445f'\"><svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'><path d='m3 10 9-7 9 7'/><path d='M5 9v11h14V9'/><path d='M9 20v-6h6v6'/></svg></a><span style='display:grid;place-items:center;width:30px;height:27px;border-radius:7px;background:#ee4655;color:white;font-size:14px'>▶</span><span style='font-family:Manrope;font-weight:800;color:#f0f5ff'>Trend Intelligence<br><small style='font:10px DM Sans;color:#91a5bc'>PREDICT · ANALYZE · RECOMMEND</small></span></div>", unsafe_allow_html=True)
    data_path = st.sidebar.text_input("Dataset path", value=str(DATA_PATH), help="Point this to the INvideos.csv file.")
    upload = st.sidebar.file_uploader("Or upload a CSV", type=["csv"], help="An uploaded CSV overrides the path for this session.")
    using_demo = False
    if upload is not None:
        try:
            raw = pd.read_csv(upload, low_memory=False)
            from youtube_analytics.data import clean_and_engineer
            frame, quality = clean_and_engineer(raw)
            quality["source_name"] = upload.name
        except Exception as error:
            st.sidebar.error(f"Could not read uploaded CSV: {error}")
            frame, quality = make_demo_data()
            using_demo = True
    else:
        try:
            source_path = Path(data_path)
            frame, quality = get_dataset(str(source_path), source_path.stat().st_mtime)
            quality["source_name"] = source_path.name
        except Exception as error:
            LOGGER.exception("Primary data load failed")
            st.sidebar.error(f"Dataset load failed: {error}")
            frame, quality = make_demo_data()
            using_demo = True
    if using_demo:
        st.sidebar.warning("Showing generated demo data because the selected CSV could not be loaded. No real-data claims are made.")
    if st.sidebar.button("Persist cleaned rows to SQLite", width="stretch"):
        try:
            with st.spinner("Writing cleaned rows and indexes to SQLite…"):
                persist_dataset(frame)
            st.sidebar.success("Cleaned dataset stored in artifacts/youtube_analytics.sqlite")
        except Exception as error:
            LOGGER.exception("SQLite persistence failed")
            st.sidebar.error(f"Could not persist dataset: {error}")

    requested_page = st.query_params.get("page")
    if requested_page in NAV_PAGES:
        st.session_state["nav_page"] = requested_page
    page = st.sidebar.radio("WORKSPACE", NAV_PAGES, index=0, key="nav_page", on_change=sync_page_query)
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(frame):,} clean rows · {frame['category_name'].nunique():,} categories · {frame['channel_title'].nunique():,} channels")
    if "country" not in frame.columns:
        st.sidebar.caption("Geographic maps unavailable: no country field in the source.")

    bundle = st.session_state.get("trained_bundle")
    if bundle is None:
        try:
            stamp = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else 0.0
            bundle = get_saved_bundle(str(MODEL_PATH), stamp)
        except Exception as error:
            st.sidebar.warning(f"Saved model could not be loaded: {error}")
            LOGGER.exception("Saved model load failed")
    filtered = global_filters(frame, bundle)
    draw_page(page, filtered, frame, quality, bundle, "country" in frame.columns and frame["country"].notna().any())
    st.markdown("<div style='margin-top:28px;padding-top:10px;border-top:1px solid #1b2d40;color:#71859d;font-size:10px'>YouTube Trend Intelligence · All forecasts are estimates, not guarantees · Source: INvideos.csv</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        LOGGER.exception("Unhandled dashboard error")
        st.error(f"The dashboard encountered an unexpected error: {error}")
        st.exception(error)