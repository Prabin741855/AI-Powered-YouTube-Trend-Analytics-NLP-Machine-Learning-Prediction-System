-- Run after `scripts/prepare_data.py`; the videos table is created from cleaned features.
SELECT
  category_name,
  COUNT(*) AS video_count,
  SUM(views) AS total_views,
  AVG(trending) AS trend_rate,
  AVG(engagement_rate) AS mean_engagement
FROM videos
GROUP BY category_name
ORDER BY total_views DESC;

SELECT
  channel_title,
  COUNT(*) AS video_count,
  AVG(views) AS mean_views,
  AVG(trending) AS trend_rate
FROM videos
GROUP BY channel_title
HAVING COUNT(*) >= 5
ORDER BY mean_views DESC;

SELECT
  published_weekday,
  published_hour,
  COUNT(*) AS video_count,
  AVG(trending) AS trend_rate,
  AVG(engagement_rate) AS mean_engagement
FROM videos
GROUP BY published_weekday, published_hour
ORDER BY trend_rate DESC;

SELECT created_at, title, category, trend_probability, expected_views,
       engagement_rate, trend_duration_days
FROM prediction_history
ORDER BY created_at DESC;