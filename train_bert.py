from google.colab import drive
drive.mount('/content/drive')

#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import nltk
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('/content/drive/My Drive/Preprocessed_Data.csv')
#Basic overview of the dataset
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

#Check for missing values and duplicates
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

#Category distribution
print("\nCategory Distribution:")
print(df['Category'].value_counts())
df = df.drop_duplicates()

# Visualize the category distribution
plt.figure(figsize=(12, 6))
sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index)
plt.title('Distribution of Job Titles (Categories)')
plt.show()

# Analyze Common Words by Category
from nltk.corpus import stopwords

# Step 1: Set up stopwords
stop_words = set(stopwords.words('english'))

# Step 2: Group data by category and join all text for each category
category_texts = df.groupby('Category')['Text'].apply(lambda texts: ' '.join(texts)).reset_index()

# Step 3: Define a function to get the most common words for a category (excluding stopwords)
def get_top_words(text, n=10):
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    common_words = Counter(words).most_common(n)
    return common_words

# Step 4: Apply the function to each category and display top 10 words
for _, row in category_texts.iterrows():
    category = row['Category']
    text = row['Text']
    top_words = get_top_words(text, n=10)
    print(f"\nTop 10 Words for Category: {category}")
    for word, freq in top_words:
        print(f"{word}: {freq}")

"""### Fine-Tuning BERT for Category Prediction"""

#!pip install datasets transformers scikit-learn torch pandas

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn import preprocessing

df = pd.read_csv('/content/drive/My Drive/Preprocessed_Data.csv')
df = df.drop_duplicates()
df_sampled = df.sample(n=1000, random_state=42)

# Label Encoding for Categories
le = preprocessing.LabelEncoder()
df_sampled["label"] = le.fit_transform(df_sampled["Category"])  # Encode categories into numeric labels
num_labels = len(le.classes_)
num_labels

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df_sampled)

# Train-test split
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
print(f"Training Dataset Size: {len(train_dataset)}")
print(f"Test Dataset Size: {len(test_dataset)}")

#distilbert-base-uncased
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
model_id = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(le.classes_),  # Number of unique categories
)

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(
        examples["Text"],
        padding="max_length",
        truncation=True,
        max_length=128,  # You can adjust the max_length as needed
    )

# Rename 'Category' to 'labels' and make sure the column 'label' exists
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set Format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

import os
os.environ["WANDB_MODE"] = "disabled"

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=0.0001,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to=None, # Disable wandb logging
)

# Define Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

"""Epoch	Training Loss	Validation Loss


1	3.776600	3.767606

2	3.693700	3.746941
"""

# Example Prediction
sample_text = "Experienced software engineer with expertise in Python and AI."
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    max_length=128,
    truncation=True,
    padding="max_length",
)
inputs = {key: value.to("cuda") for key, value in inputs.items()}
model.to("cuda")
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_label_id = torch.argmax(logits, dim=-1).item()
    predicted_label = le.inverse_transform([predicted_label_id])[0]

print(f"Predicted Category: {predicted_label}")