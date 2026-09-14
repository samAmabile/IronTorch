from datasets import load_dataset
import pandas as pd
import types 

ds = load_dataset("stanfordnlp/imdb", split="train")

df = ds.to_pandas()

pos = df[df["label"] == 1]
neg = df[df["label"] == 0]

pos5k = pos[:10000]
neg5k = neg[:10000]

saveas = "data/reviews8.csv"

print(len(pos5k), len(neg5k))

combined = pd.concat([pos5k, neg5k], ignore_index=True)


combined["text"] = combined["text"].str.lower()
combined["text"] = combined["text"].str.replace(r"<br\s*/?>", " ", regex=True)
combined["text"] = combined["text"].str.replace(r"\s+", " ", regex=True).str.strip()

combined.to_csv(saveas, index=False)



