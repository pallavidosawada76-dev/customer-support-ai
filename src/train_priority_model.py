from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================
# 1. LOAD DATASET
# ============================================

data = load_dataset(
    "Prady06/customer-support-tickets",
    split="train"
)


# ============================================
# 2. FILTER ENGLISH DATA
# ============================================

data = data.filter(
    lambda row: row["language"] == "en"
)


print("Total English tickets:", len(data))


# ============================================
# 3. CREATE TEXT INPUT
# ============================================

X = []
y = []


for row in data:

    subject = row["subject"] or ""
    body = row["body"] or ""

    # Combine subject and body
    text = subject + " " + body

    # Ignore tickets with no usable text
    if text.strip() == "":
        continue

    X.append(text)
    y.append(row["priority"])


print("Usable tickets:", len(X))


# ============================================
# 4. PRIORITY DISTRIBUTION
# ============================================

from collections import Counter

print("\n========== PRIORITY DISTRIBUTION ==========")

priority_counts = Counter(y)

for priority, count in priority_counts.items():
    print(priority, ":", count)


# ============================================
# 5. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n========== TRAIN / TEST SPLIT ==========")

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))


# ============================================
# 6. TF-IDF
# ============================================

vectorizer = TfidfVectorizer(
    max_features=10000
)


X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\n========== TF-IDF ==========")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# ============================================
# 7. TRAIN PRIORITY MODEL
# ============================================

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_tfidf,
    y_train
)


print("\n========== PRIORITY MODEL ==========")

print("Priority model trained successfully!")


# ============================================
# 8. MAKE PREDICTIONS
# ============================================

y_pred = model.predict(X_test_tfidf)


# ============================================
# 9. EVALUATION
# ============================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========== MODEL EVALUATION ==========")

print("Priority Accuracy:", accuracy)


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================
# 10. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted Priority")
plt.ylabel("Actual Priority")

plt.title("Priority Confusion Matrix")

plt.show()


# ============================================
# 11. TEST NEW TICKETS
# ============================================

new_tickets = [

    "I forgot my password and need help accessing my account",

    "My payment failed and I cannot complete my purchase",

    "Our entire company's account system is down and nobody can access their accounts",

    "Can you tell me what payment methods you accept?",

    "I would like some information about your product features"
]


# Convert new tickets using SAME vectorizer

new_tickets_tfidf = vectorizer.transform(
    new_tickets
)


# Predict priority

predictions = model.predict(
    new_tickets_tfidf
)


print("\n========== NEW PRIORITY PREDICTIONS ==========")


for ticket, prediction in zip(
    new_tickets,
    predictions
):

    print("\nTicket:", ticket)

    print("Predicted priority:", prediction)