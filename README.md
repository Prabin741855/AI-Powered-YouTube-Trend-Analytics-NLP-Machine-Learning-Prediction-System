<img width="1536" height="1024" alt="ChatGPT Image Sep 28, 2026, 06_49_26 PM" src="https://github.com/user-attachments/assets/ef7122b6-59df-4d54-b8ab-110adb4dd7ed" />

# AI-Powered-YouTube-Trend-Analytics-NLP-Machine-Learning-Prediction-System
YouTube Trend Analytics & NLP is a Python-based Data Science and Machine Learning project that analyzes 50,000 YouTube video records from INvideos.csv. The system performs data cleaning, trend analysis, engagement analysis, NLP on video titles/descriptions/tags, sentiment analysis, keyword extraction, topic analysis, and ML-based trend prediction through an interactive Streamlit dashboard.

🛠️ Tools & Technologies

Python | Pandas | NumPy | Scikit-learn | NLTK | spaCy | XGBoost | Plotly | Matplotlib | Seaborn | WordCloud | Streamlit | SHAP | Jupyter Notebook | Git | GitHub

📊 Main Features
Trend Analytics: trending videos, trend duration, views and growth patterns
Engagement Analysis: views, likes, comments and engagement rate
NLP Analysis: title, description and tag processing
Sentiment Analysis: positive, neutral and negative content
Keyword Analysis: TF-IDF and frequent keywords
Topic Analysis: identify major video topics
Word Cloud: visualize frequently used words
Channel Analysis: channel performance and trending activity
Category Analysis: category-wise trends and engagement
ML Prediction: predict trending status using machine-learning models
Interactive Filters: date, category, channel, views, trending status and keywords
Dashboard: interactive charts, KPIs, tables and insights
🔄 Workflow
INvideos.csv
     ↓
Data Cleaning & Validation
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
NLP Processing
     ↓
Trend & Engagement Analysis
     ↓
Machine Learning
     ↓
Prediction & Insights
     ↓
Interactive Streamlit Dashboard
📦 Installation
python -m venv .venv

Windows:

.venv\Scripts\activate

Install dependencies:

pip install pandas numpy scikit-learn nltk spacy xgboost plotly matplotlib seaborn wordcloud streamlit shap jupyter joblib

Install NLP resources:

python -m nltk.downloader punkt stopwords wordnet omw-1.4
python -m spacy download en_core_web_sm
📁 Dataset

Place the dataset in:

project/
├── data/
│   └── INvideos.csv
├── app.py
├── requirements.txt
└── README.md
▶️ Run
streamlit run app.py

Open:

http://localhost:8501

GitHub Short Description:

An interactive YouTube Trend Analytics and NLP project using Python, Machine Learning, and Streamlit to analyze 50K videos, discover trends, extract keywords and topics, perform sentiment analysis, and predict trending content.
