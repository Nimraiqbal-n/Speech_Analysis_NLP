# Step 1: Install dependencies
!pip install transformers torch pandas matplotlib

# Step 2: Import libraries
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

# Step 3: Upload speech files (imran.txt, cm.txt)
print("Upload imran.txt and cm.txt files...")
uploaded = files.upload()

# Step 4: Read speeches
with open("imran.txt", "r", encoding="utf-8") as f:
    imran_speech = f.readlines()

with open("cm.txt", "r", encoding="utf-8") as f:
    cm_speech = f.readlines()

# Step 5: Load sentiment model
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Label mapping (Roberta model gives LABEL_0, LABEL_1, LABEL_2)
label_map = {
    "LABEL_0": "Toxic",        # Negative
    "LABEL_1": "Neutral",      # Neutral
    "LABEL_2": "Non-Toxic"     # Positive
}

# Step 6: Run classification
data = []

# Imran Khan speeches
for line in imran_speech:
    if line.strip():
        result = classifier(line)[0]
        label = label_map[result['label']]
        data.append(("Imran Khan", line.strip(), label, result['score']))

# CM Punjab speeches
for line in cm_speech:
    if line.strip():
        result = classifier(line)[0]
        label = label_map[result['label']]
        data.append(("CM Punjab", line.strip(), label, result['score']))

# Step 7: Put into DataFrame
df = pd.DataFrame(data, columns=["Speaker", "Text", "Label", "Confidence"])
print("Sample Results:")
print(df.head())

# Step 8: Summarize proportions
summary = df.groupby("Speaker")["Label"].value_counts(normalize=True).unstack().fillna(0)

# Step 9: Plot comparison chart
summary.plot(kind="bar", stacked=True, figsize=(8,6))
plt.title("Speech Sentiment Comparison: Imran Khan vs CM Punjab")
plt.ylabel("Proportion of Speech")
plt.show()

