import pandas as pd
import numpy as np
import re
import joblib
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from nomic import embed
from mlflow import sklearn as mlflow_sklearn

class RacistContentClassifierTrainer:
    def __init__(self, data_path: str, model_path: str, random_state: int = 42):
        self.data_path = data_path
        self.model_path = model_path
        self.random_state = random_state

        mlflow.set_experiment("Racist Content Social Media")

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_path, usecols=['Cleaned tweet', 'Tag'])
            df.columns = ['text', 'label']
            df['text'] = df['text'].apply(self.remove_non_ascii)
            return df
        except Exception as e:
            mlflow.log_artifact(f"Data loading failed: {e}")
            raise

    @staticmethod
    def remove_non_ascii(text: str) -> str:
        return re.sub(r'[^\x00-\x7F]+', '', text)

    def train(self):
        try:
            df = self.load_data()
            output = embed.text(texts=df['text'].tolist(), model='nomic-embed-text-v1')
            embeddings = np.array(output['embeddings'])
            X_train, X_test, y_train, y_test = train_test_split(embeddings, df['label'], test_size=0.2, random_state=self.random_state)

            param_grid = {
                'n_estimators': [200, 400, 600], 
                'max_depth': [None, 10, 20, 30],  
                'min_samples_split': [2, 5, 10, 15],  
                'min_samples_leaf': [1, 2, 4]
            }

            with mlflow.start_run():
                rf_model = RandomForestClassifier(random_state=self.random_state)
                grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
                grid_search.fit(X_train, y_train)

                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("best_score", grid_search.best_score_)

                self.model = grid_search.best_estimator_

                mlflow_sklearn.log_model(self.model, "RandomForestClassifier")
                mlflow.log_metric("Best Score", grid_search.best_score_)

                score_message = f"Training complete. Best Score: {grid_search.best_score_}"
                with open("score_message.txt", "a") as f:
                    f.write(score_message + "\n") 
                mlflow.log_artifact("score_message.txt")

        except Exception as e:
            mlflow.log_artifact(f"Model training failed: {e}")
            raise

    def save_model(self):
        if self.model:
            joblib.dump(self.model, self.model_path)
            mlflow.log_artifact(self.model_path, "Saved Models")
        else:
            mlflow.log_artifact("Model not trained.")

if __name__ == "__main__":
    data_path = '../experiments/data/tweets.csv'
    model_path = './rf_model.joblib'
    trainer = RacistContentClassifierTrainer(data_path=data_path, model_path=model_path)
    trainer.train()
    trainer.save_model()
    print("Model trained and saved.")
