from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from youtube_analytics.config import CATEGORY_METADATA_PATH, LOGGER


NUMERIC_COLUMNS = ["views", "likes", "dislikes", "comments", "duration_seconds", "thumbnail_score", "trend_days"]
TEXT_COLUMNS = ["title", "description", "tags", "channel_title", "category_name"]
MODEL_NUMERIC_FEATURES = [
    "title_length", "title_word_count", "description_word_count", "hashtag_count", "tag_count",
    "sentiment_score", "published_hour", "published_weekday", "published_month", "duration_seconds",
    "channel_prior_videos", "channel_prior_trend_rate", "channel_prior_median_views",
    "category_prior_trend_rate", "category_prior_videos", "thumbnail_score",
]

POSITIVE_WORDS = {
    "amazing", "best", "better", "breakthrough", "excellent", "fun", "funny", "good", "great",
    "guide", "helpful", "incredible", "learn", "love", "new", "perfect", "review", "top", "win", "wow",
}
NEGATIVE_WORDS = {
    "bad", "boring", "fail", "hate", "issue", "mistake", "negative", "problem", "sad", "worst", "wrong",
}


def _sentiment(text: str) -> float:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return 0.0
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    return round((positive - negative) / max(len(tokens), 1), 4)


def clean_and_engineer(raw: pd.DataFrame, category_metadata_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize source fields and compute features available at publish time."""
    frame = raw.copy()
    report: dict[str, Any] = {"input_rows": int(len(frame)), "input_columns": int(len(frame.columns))}
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    required = {"title", "published_at", "views", "likes", "comments"}
    missing_required = required - set(frame.columns)
    if missing_required:
        raise ValueError(f"Missing required dataset columns: {', '.join(sorted(missing_required))}")

    for column in TEXT_COLUMNS:
        if column not in frame:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    if "video_id" not in frame:
        frame["video_id"] = frame["title"].str.lower() + "|" + frame["published_at"].astype(str)
    frame["video_id"] = frame["video_id"].fillna("").astype(str)
    before_duplicates = len(frame)
    frame = frame.drop_duplicates("video_id", keep="last")
    report["duplicate_rows_removed"] = int(before_duplicates - len(frame))

    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce", utc=True).dt.tz_convert(None)
    invalid_dates = int(frame["published_at"].isna().sum())
    empty_titles = int(frame["title"].str.len().eq(0).sum())
    frame = frame.dropna(subset=["published_at"])
    frame = frame.loc[frame["title"].str.len().gt(0)].copy()
    report["invalid_dates_removed"] = invalid_dates
    report["empty_titles_removed"] = empty_titles

    for column in NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median = frame[column].median()
        frame[column] = frame[column].fillna(0 if pd.isna(median) else median).clip(lower=0)

    if "trending" not in frame:
        frame["trending"] = 0
    frame["trending"] = pd.to_numeric(frame["trending"], errors="coerce").fillna(0).gt(0).astype("int8")
    if "category_id" not in frame:
        frame["category_id"] = -1
    frame["category_id"] = pd.to_numeric(frame["category_id"], errors="coerce").fillna(-1).astype(int)

    metadata_path = category_metadata_path or CATEGORY_METADATA_PATH
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mapping = {int(key): str(value) for key, value in metadata.items()}
            mapped = frame["category_id"].map(mapping)
            frame["category_name"] = mapped.fillna(frame["category_name"]).replace("", "Unknown")
            report["category_metadata"] = str(metadata_path.name)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            LOGGER.warning("Could not load category metadata: %s", error)
            report["category_metadata_warning"] = str(error)
    frame["category_name"] = frame["category_name"].replace("", "Unknown").fillna("Unknown")

    frame["published_hour"] = frame["published_at"].dt.hour.astype("int16")
    frame["published_weekday"] = frame["published_at"].dt.dayofweek.astype("int16")
    frame["published_month"] = frame["published_at"].dt.month.astype("int16")
    frame["published_date"] = frame["published_at"].dt.date
    frame["title_length"] = frame["title"].str.len().astype("int32")
    frame["title_word_count"] = frame["title"].str.findall(r"\b\w+\b").str.len().astype("int16")
    frame["description_word_count"] = frame["description"].str.findall(r"\b\w+\b").str.len().astype("int32")
    frame["hashtag_count"] = (frame["title"] + " " + frame["description"] + " " + frame["tags"]).str.count(r"(?<!\w)#\w+").astype("int16")
    frame["tag_count"] = frame["tags"].str.split(",").map(lambda tags: len([tag for tag in tags if tag.strip()])).astype("int16")
    frame["sentiment_score"] = (frame["title"] + " " + frame["description"]).map(_sentiment).astype("float32")
    frame["engagement_rate"] = (frame["likes"] + frame["comments"]) / frame["views"].replace(0, np.nan)
    frame["like_rate"] = frame["likes"] / frame["views"].replace(0, np.nan)
    frame["comment_rate"] = frame["comments"] / frame["views"].replace(0, np.nan)
    frame[["engagement_rate", "like_rate", "comment_rate"]] = frame[["engagement_rate", "like_rate", "comment_rate"]].replace([np.inf, -np.inf], np.nan).fillna(0)
    frame["views_log"] = np.log1p(frame["views"])
    frame["trend_duration_days"] = frame["trend_days"].astype("float32")

    frame = frame.sort_values(["published_at", "video_id"], kind="mergesort").reset_index(drop=True)
    channel_groups = frame.groupby("channel_title", sort=False, dropna=False)
    frame["channel_prior_videos"] = channel_groups.cumcount().astype("int32")
    frame["_channel_trend_cum"] = channel_groups["trending"].cumsum() - frame["trending"]
    frame["channel_prior_trend_rate"] = np.divide(
        frame["_channel_trend_cum"], frame["channel_prior_videos"],
        out=np.zeros(len(frame), dtype=float), where=frame["channel_prior_videos"].to_numpy() > 0,
    )
    frame["channel_prior_median_views"] = channel_groups["views"].transform(lambda values: values.shift(1).expanding().median()).fillna(0)
    category_groups = frame.groupby("category_name", sort=False, dropna=False)
    frame["category_prior_videos"] = category_groups.cumcount().astype("int32")
    frame["_category_trend_cum"] = category_groups["trending"].cumsum() - frame["trending"]
    frame["category_prior_trend_rate"] = np.divide(
        frame["_category_trend_cum"], frame["category_prior_videos"],
        out=np.zeros(len(frame), dtype=float), where=frame["category_prior_videos"].to_numpy() > 0,
    )
    frame = frame.drop(columns=["_channel_trend_cum", "_category_trend_cum"])

    for column in ["views", "likes", "dislikes", "comments"]:
        q1, q3 = frame[column].quantile([0.25, 0.75])
        upper = q3 + 3 * (q3 - q1)
        frame[f"{column}_outlier"] = frame[column].gt(upper) if upper > 0 else False

    report.update({
        "rows_after_cleaning": int(len(frame)),
        "missing_cells_after_cleaning": int(frame.isna().sum().sum()),
        "date_min": str(frame["published_at"].min().date()) if not frame.empty else "",
        "date_max": str(frame["published_at"].max().date()) if not frame.empty else "",
        "trending_videos": int(frame["trending"].sum()),
        "category_count": int(frame["category_name"].nunique()),
        "channel_count": int(frame["channel_title"].nunique()),
        "outlier_flag_counts": {f"{column}_outlier": int(frame[f"{column}_outlier"].sum()) for column in ["views", "likes", "dislikes", "comments"]},
    })
    LOGGER.info("Cleaned %s rows to %s rows", report["input_rows"], report["rows_after_cleaning"])
    return frame, report


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw = pd.read_csv(path, low_memory=False)
    return clean_and_engineer(raw)


def make_prediction_row(
    title: str,
    description: str,
    category_name: str,
    published_at: pd.Timestamp,
    duration_seconds: float,
    thumbnail_score: float,
    historical: pd.DataFrame,
    channel_title: str = "New channel",
    tags: str = "",
) -> pd.DataFrame:
    """Create inference features from upload metadata and historical priors only."""
    category_rows = historical.loc[historical["category_name"].eq(category_name)]
    channel_rows = historical.loc[historical["channel_title"].eq(channel_title)]
    channel_trend_rate = float(channel_rows["trending"].mean()) if not channel_rows.empty else float(historical["trending"].mean())
    channel_prior_views = float(channel_rows["views"].median()) if not channel_rows.empty else float(historical["views"].median())
    full_text = f"{title} {description} {tags}"
    title_tokens = re.findall(r"\b\w+\b", title)
    description_tokens = re.findall(r"\b\w+\b", description)
    tag_count = len([item for item in tags.split(",") if item.strip()])
    row = {
        "title": title,
        "title_text": title,
        "category_name": category_name,
        "title_length": len(title),
        "title_word_count": len(title_tokens),
        "description_word_count": len(description_tokens),
        "hashtag_count": len(re.findall(r"(?<!\w)#\w+", full_text)),
        "tag_count": tag_count,
        "sentiment_score": _sentiment(f"{title} {description}"),
        "published_hour": published_at.hour,
        "published_weekday": published_at.dayofweek,
        "published_month": published_at.month,
        "duration_seconds": max(float(duration_seconds), 0),
        "channel_prior_videos": len(channel_rows),
        "channel_prior_trend_rate": channel_trend_rate,
        "channel_prior_median_views": channel_prior_views,
        "category_prior_trend_rate": float(category_rows["trending"].mean()) if not category_rows.empty else float(historical["trending"].mean()),
        "category_prior_videos": len(category_rows),
        "thumbnail_score": float(thumbnail_score),
    }
    return pd.DataFrame([row])


def topic_analysis(texts: pd.Series, topics: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return document-term frequency and topic keyword tables from TF-IDF/NMF."""
    from sklearn.decomposition import NMF

    corpus = texts.fillna("").astype(str)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=4000, min_df=2, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus)
    frequency = np.asarray(matrix.sum(axis=0)).ravel()
    terms = np.asarray(vectorizer.get_feature_names_out())
    keywords = pd.DataFrame({"term": terms, "weight": frequency}).nlargest(30, "weight")
    topic_count = max(1, min(topics, matrix.shape[0] - 1, matrix.shape[1] - 1))
    nmf = NMF(n_components=topic_count, init="nndsvda", random_state=42, max_iter=250)
    nmf.fit(matrix)
    topic_rows = []
    for topic_id, weights in enumerate(nmf.components_):
        top_indices = weights.argsort()[::-1][:8]
        topic_rows.extend({"topic": f"Topic {topic_id + 1}", "term": terms[index], "weight": float(weights[index])} for index in top_indices)
    return keywords, pd.DataFrame(topic_rows)
