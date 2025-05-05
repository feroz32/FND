import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# Load data
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add labels
true_df['label'] = 1
fake_df['label'] = 0

# Balance dataset
min_len = min(len(true_df), len(fake_df))
true_df = true_df.sample(n=min_len, random_state=42)
fake_df = fake_df.sample(n=min_len, random_state=42)

# Combine and shuffle
df = pd.concat([true_df, fake_df])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['title'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Save model & vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
