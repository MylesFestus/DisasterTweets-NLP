# Natural Language Processing with Disaster Tweets

## Predict which Tweets are about real disasters and which ones are not

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)


***Acknowledgments***

This dataset was created by the company figure-eight and originally shared on their ‘Data For Everyone’ website here.

***Project Overview***

This project develops a Natural Language Processing (NLP) pipeline to classify tweets as either **disaster-related** or **non-disaster-related**. The objective is to build and evaluate multiple machine learning models using different text vectorization techniques and identify the best-performing approach for disaster tweet detection.

The project follows a complete machine learning workflow, including text preprocessing, feature engineering, model comparison, hyperparameter tuning, and final model evaluation.

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

***1. Text Preprocessing***

The following preprocessing steps were applied:

- Lowercasing text
- Removing URLs
- Removing punctuation
- Tokenization
-  Stopword removal
- Lemmatization
- Text reconstruction for vectorization

***Example***

***Original Tweet***

Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all

***Processed Tweet***

deed reason earthquake may allah forgive u

***2. Feature Engineering***

Two vectorization techniques were evaluated:

***Count Vectorizer***

Converts text into a matrix of word occurrence counts.

***TF-IDF Vectorizer***

Computes term importance by weighting words based on frequency within and across documents.

***Models Evaluated***

**Baseline Models**
- Logistic Regression
- Decision Tree Classifier
- Linear Support Vector Classifier (Linear SVC)

**Ensemble Models**

- Random Forest Classifier
- XGBoost Classifier

***Model Evaluation***

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

***Primary Metric***

**F1-Score** was selected because it balances precision and recall and is suitable for binary classification tasks where class distributions may not be perfectly balanced.

***Hyperparameter Tuning***

GridSearchCV was used to optimize:

***Vectorizer Parameters***
- max_features
- ngram_range

**Logistic Regression Parameters**
- Regularization strength (C)

**Linear SVC Parameters**
- Regularization strength (C)

Cross-validation was performed using 5-fold CV to identify the optimal parameter combination.

***Project Structure***

DisasterTweets-NLP/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── disaster_tweets.ipynb
│
├── src/
│
├── models/
│   └── best_disaster_tweet_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore


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

***Running the Project***

Launch Jupyter Notebook:

jupyter notebook

Execute the notebook cells to:

1. Preprocess text
2. Generate vectorized features
3. Train machine learning models
4. Compare model performance
5. Tune hyperparameters
6. Save the best-performing model

***Results***

The project compares multiple machine learning algorithms and vectorization techniques to identify the best-performing disaster tweet classifier.

Typical findings include:

- TF-IDF often outperforms Count Vectorizer.
- Linear SVC and Logistic Regression generally achieve the highest F1-scores.
- Tree-based models tend to underperform on sparse text data.
- Hyperparameter tuning improves model generalization.

Update this section with your final model performance metrics after training.

***Model Persistence***

Save the best model:

import joblib

joblib.dump(best_model, "best_disaster_tweet_model.pkl")

Load the saved model:

import joblib

model = joblib.load("best_disaster_tweet_model.pkl")

***Future Improvements***

- Deep Learning (LSTM, GRU)
- Transformer Models (BERT, RoBERTa)
- Model Explainability using SHAP
- Ensemble Stacking
- Real-time Tweet Classification API
- Deployment using Streamlit or FastAPI

***Notable libraries Used***
- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- NLTK
- Joblib

***Author***

Myles Festus

Natural Language Processing | Machine Learning | Data Science

GitHub: https://github.com/MylesFestus
