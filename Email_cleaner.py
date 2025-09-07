import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("large_emails_dataset.csv")
df["text"] = df["subject"] + " " + df["body"]

X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Evaluate
print("Evaluation Report:\n")
print(classification_report(y_test, model.predict(X_test)))

# Classify new email
while True:
    print("\n--- Classify a New Email ---")
    subject = input("Enter subject: ")
    body = input("Enter body: ")
    text = subject + " " + body
    prediction = model.predict([text])[0]
    print(f"🧹 Suggested Label: {prediction}")
    if prediction in ["Spam", "Promotions"]:
        print("✅ Suggestion: Delete")
    else:
        print("📌 Suggestion: Keep")
