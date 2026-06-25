# Natural Language Processing with Disaster Tweets

## Predict which Tweets are about real disasters and which ones are not

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)


***Acknowledgments***

This dataset was created by the company figure-eight and originally shared on their ‘Data For Everyone’ website here.

***Project Overview***

This project develops a Natural Language Processing (NLP) pipeline to classify tweets as either **disaster-related** or **non-disaster-related**. The objective is to build and evaluate multiple machine learning models using different text vectorization techniques and identify the best-performing approach for disaster tweet detection.

The project follows a complete machine learning workflow, including text preprocessing, feature engineering, model comparison, hyperparameter tuning, and final model evaluation.


***Project Structure***

DisasterTweets-NLP/
│
├── data/
│   ├─disaster_tweets_test.csv
│   └── disaster_tweets_train.csv
│
├── notebooks/
│   └── disaster_tweets.ipynb
│
├── src/
│  ├── api.py
│  ├── preprocessing.py
│  └── train.py
│  
├── models/
│   └── best_disaster_tweet_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore


***Data Set***

The dataset consists of tweets labeled as:

- 1 → Disaster-related tweet
- 0 → Non-disaster-related tweet

***Features***

Tweet source: https://twitter.com/AnyOtherAnnaK/status/629195955506708480

Kaggle source: https://www.kaggle.com/competitions/nlp-getting-started/overview

***Project Structure***

***Features***

Feature	Description
id	Unique tweet identifier
text	Tweet content
target	Classification label (0 or 1)

***Project Workflow***

1. Data Loading

2. Text Preprocessing
- Lowercasing text
- Removing URLs
- Removing punctuation
- Tokenization
- Stopword removal
- Lemmatization
- Text reconstruction for vectorization
- 
3. Text Vectorization
- Count Vectorizer --> Converts text into a matrix of word occurrence counts.
- TF-IDF Vectorizer --> Computes term importance by weighting words based on frequency within and across documents.

4. Model Selection (Simple & Ensemble models)
- LogisticRegression
- Linear SVM
- RandomForest
- DecisionTree

5. Model Evaluation
- Precision --> Of tweets predicted, how many were actually disaster?
- Recall --> Of actual disaster tweets, how many did we catch?
- F1 Score --> Penalizes low recall and low precision
- Confusion Matrix --> How well model distinguishes between disaster and non disaster tweets

6. Hyperparameter Tunning of best models

**Best Parameters found:**
- model --> LogisticRegression(max_iter=1000)
- vectorizer --> TfidfVectorizer(max_features=10000)
- vectorizer__ngram_range(1, 1)

***Results***

The project compares multiple machine learning algorithms and vectorization techniques to identify the best-performing disaster tweet classifier.

Typical findings include:

- TF-IDF often outperforms Count Vectorizer.
- Linear SVC and Logistic Regression generally achieve the highest F1-scores.
- Tree-based models tend to underperform on sparse text data.
- Hyperparameter tuning improves model generalization only slightly.

***Installation***

***Clone the Repository***

git clone git@github.com:MylesFestus/DisasterTweets-NLP.git

***Navigate to the Project Directory***

cd DisasterTweets-NLP

***Create a Virtual Environment***

python -m venv .venv

***Activate the Environment***

***macOS/Linux***

source .venv/bin/activate

***Windows***

.venv\Scripts\activate

***Install Dependencies***

pip install -r requirements.txt


***Usage***

data folder directory added to `.gitignore`

***Download data***

***Train the model***

`python src/train.py`

Saves model as `models/best_disaster_tweet_model.pkl`

***Run the API***

`ùnicorn src.api:app --reload`

***Endpoints***

- `GET /` - health check
- `POST /predict` - classify a text

`curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \ -H 'accept: application/json' \ -H 'Content-Type: application/json' \ -d '{
  "text": "There'\''s an emergency evacuation happening now in the building across the street"}'`

`{"disaster": 1, "probability": 0.9617164890467441}`


***Future Improvements***

- Deep Learning (LSTM, GRU)
- Transformer Models (BERT, RoBERTa)
- Model Explainability using SHAP
- Ensemble Stacking
- Real-time Tweet Classification API
- Deployment using Streamlit or FastAPI


***Author***

Myles Festus

Natural Language Processing | Machine Learning | Data Science

GitHub: https://github.com/MylesFestus
