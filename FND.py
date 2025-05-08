<<<<<<< HEAD
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
# Load the data (True and Fake News CSV files)
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Add label columns to differentiate between true and fake news
true_df['label'] = 1  # Label 1 for real news
fake_df['label'] = 0  # Label 0 for fake news

# Combine the datasets
df = pd.concat([true_df, fake_df])

# Shuffle the data
df = df.sample(frac=1).reset_index(drop=True)

# Only use title and text columns
df = df[['title', 'text', 'label']]

# Show the first few rows of the dataset
print(df.head())
# Text data (titles and text)
X = df['text']  # Use the 'text' column for classification
y = df['label']  # Use the 'label' column for the target

# Initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the text data into a matrix of TF-IDF features
X_vectorized = vectorizer.fit_transform(X)

# Show the shape of the vectorized text data (how many samples and features)
print(f"Shape of vectorized data: {X_vectorized.shape}")
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Initialize the model
model = PassiveAggressiveClassifier(max_iter=50)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict the labels of the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Confusion Matrix:\n{conf_matrix}")
def predict_news(news_text):
    # Transform the input news text using the same vectorizer
    input_data = vectorizer.transform([news_text])
    
    # Predict if the news is real or fake
    prediction = model.predict(input_data)
    
    # Return the result
    return "Real" if prediction[0] == 1 else "Fake"

# Test the function with a custom input
sample_news = input("Enter a news article: ")
print("Prediction:", predict_news(sample_news))
=======

>>>>>>> 4c42a1caf35af7110028626d2838d8e76f1d7036
