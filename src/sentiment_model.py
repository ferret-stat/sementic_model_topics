import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

label_map = {
    0: "neutral",
    1: "positive",
    2: "negative"
}

max_len = 200

path = os.getcwd().replace("\\src", "")
model_path = os.path.join(path, "models", "sentiment_model.keras")
tokenizer_path = os.path.join(path, "models", "tokenizer.pkl")

model = tf.keras.models.load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)


def predict_review(text, model, tokenizer, max_len=200):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probs = model.predict(padded, verbose=0)[0]
    pred_class = int(np.argmax(probs))

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_class": pred_class,
        "predicted_label": label_map[pred_class],
        "negative_prob": float(probs[2]),
        "neutral_prob": float(probs[0]),
        "positive_prob": float(probs[1]),
    }

review = """"очередь долгая , один врач на всех"""
result = predict_review(review, model, tokenizer, max_len=max_len)

print(result)