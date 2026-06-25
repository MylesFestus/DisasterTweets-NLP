import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


from preprocessing import *

nltk.download('stopwords')


def main():
    # Load dataset
    df = pd.read_csv(
        '/Users/festusattornelson/Documents/Projects/Python_Udemy/Projects/NLP-disastertweets/data/disaster_treets_train.csv')

    df = pd.DataFrame({'text': df['text'], 'target': df['target']})

    X = df['text']
    y = df['target']

    # Data Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)


    # Baseline Model Training and Evaluation
    # Defining Vectorizers
    vectorizers = {
        "CountVectorizer": CountVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        ),
        "TF-IDF": TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
    }


    # Defining models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Linear SVC": LinearSVC()
    }

    # Model Training
    results = []
    print("\nStarting Baseline Model Training...")
    best_f1_score = 0
    best_baseline_model = None
    best_baseline_vectorizer_name = None

    for vec_name, vectorizer in vectorizers.items():
        print(f"\nApplying {vec_name}...")
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        for model_name, model in models.items():
            print(f"Training {model_name} with {vec_name}...")
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            results.append({
                "Vectorizer": vec_name,
                "Model": model_name,
                "Accuracy": accuracy,
                "F1_Score": f1
            })

            if f1 > best_f1_score:
                best_f1_score = f1
                best_baseline_model = model_name
                best_baseline_vectorizer_name = vec_name

    results_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False)
    best_baseline = results_df.iloc[0]
    print("\nBaseline Model Results:")
    print(results_df)

    # --- Hyperparameter Tuning ---
    print("\nHyperparameter Tuning with GridSearchCV...")
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("model", LogisticRegression())
    ])

    param_grid = [
        {
            "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "vectorizer__max_features": [5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "model": [LogisticRegression(max_iter=1000)],
            "model__C": [0.01, 0.1, 1, 10]
        },
        {
            "vectorizer": [CountVectorizer(), TfidfVectorizer()],
            "vectorizer__max_features": [5000, 10000],
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "model": [LinearSVC()],
            "model__C": [0.01, 0.1, 1, 10]
        }
    ]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=0  # Suppress verbose output during grid search for cleaner script execution
    )

    grid_search.fit(X_train, y_train)

    print("\nHyperparameter Tuning Complete.")
    print("Best parameters found: ", grid_search.best_params_)
    print("Best CV F1 Score: ", grid_search.best_score_)


    # --- Tuned Model Evaluation ---
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)

    final_accuracy = accuracy_score(y_test, y_pred_tuned)
    final_f1 = f1_score(y_test, y_pred_tuned)

    print("\n--- Tuned Model Performance ---")
    print("Accuracy :", final_accuracy)
    print("F1 Score :", final_f1)
    print("\nClassification Report")
    print(classification_report(y_test, y_pred_tuned))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred_tuned))

    # --- Comparison ---
    print("\n--- Performance Comparison ---")
    print("Best Baseline Model:")
    print(f"  Vectorizer: {best_baseline['Vectorizer']}")
    print(f"  Model: {best_baseline['Model']}")
    print(f"  Accuracy: {best_baseline['Accuracy']:.4f}")
    print(f"  F1 Score: {best_baseline['F1_Score']:.4f}")

    print("\nHyperparameter Tuned Model:")
    print(f"  Accuracy: {final_accuracy:.4f}")
    print(f"  F1 Score: {final_f1:.4f}")


    # save model






if __name__ == "__main__":
    main()



