import numpy as np
import joblib
from nomic import embed


class RacistContentClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, text: str) -> str:
        input_embedding = embed.text(texts=[text], model="nomic-embed-text-v1")
        input_embedding = np.array(input_embedding["embeddings"])
        prediction = self.model.predict(input_embedding)
        return "Racist" if prediction[0] == 1 else "Not racist"
