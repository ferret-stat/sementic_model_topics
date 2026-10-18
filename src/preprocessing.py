import re
import pandas as pd

    
def clean_text(text):
    
    text = str(text).lower()
    
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^а-яё\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

def preprocessing(path: str):
    df = pd.read_csv(path)
    df["clean_review"] = df["text"].apply(clean_text)
    
    return df

def main():
    print(preprocessing("data/datasets.csv"))

if __name__ == "__main__":
    main()