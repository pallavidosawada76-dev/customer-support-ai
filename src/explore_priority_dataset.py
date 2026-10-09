from datasets import load_dataset
from collections import Counter


# ============================================
# 1. LOAD DATASET
# ============================================

data = load_dataset(
    "Prady06/customer-support-tickets",
    split="train"
)


# ============================================
# 2. FILTER ENGLISH TICKETS
# ============================================

english_data = data.filter(
    lambda row: row["language"] == "en"
)


print("========== ENGLISH DATA ==========")

print("Total English tickets:", len(english_data))


# ============================================
# 3. PRIORITY DISTRIBUTION
# ============================================

print("\n========== ENGLISH PRIORITY DISTRIBUTION ==========")

priority_counts = Counter(english_data["priority"])

for priority, count in priority_counts.items():
    print(priority, ":", count)


# ============================================
# 4. MISSING VALUES
# ============================================

print("\n========== MISSING VALUES ==========")

for column in ["subject", "body", "priority"]:

    missing = sum(
        1
        for value in english_data[column]
        if value is None or value == ""
    )

    print(column, ":", missing)


# ============================================
# 5. SAMPLE ENGLISH TICKETS
# ============================================

print("\n========== SAMPLE ENGLISH TICKETS ==========")

count = 0

for row in english_data:

    if row["subject"] and row["body"]:

        print("\nSubject:", row["subject"])
        print("Body:", row["body"])
        print("Priority:", row["priority"])

        count += 1

        if count == 5:
            break