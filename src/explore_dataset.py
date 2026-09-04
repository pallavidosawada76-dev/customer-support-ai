from datasets import load_dataset

dataset = load_dataset(
    "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
)

train_data = dataset["train"]

print("Number of rows:", len(train_data))

print("\nColumn names:")
print(train_data.column_names)

print("\nFirst row:")
print(train_data[0])

from collections import Counter

categories = [row["category"] for row in train_data]
intents = [row["intent"] for row in train_data]

print("\nNumber of unique categories:", len(set(categories)))
print("Categories:")
print(set(categories))

print("\nNumber of unique intents:", len(set(intents)))

print("\nIntent distribution:")
print(Counter(intents))


import random

print("\nRandom examples:")

for i in range(5):
    index = random.randint(0, len(train_data) - 1)
    print("\n--- Example", i + 1, "---")
    print("Instruction:", train_data[index]["instruction"])
    print("Category:", train_data[index]["category"])
    print("Intent:", train_data[index]["intent"])
    print("Response:", train_data[index]["response"])


    print("\n========== DATA QUALITY CHECK ==========")

columns = ["instruction", "category", "intent", "response"]

for column in columns:
    missing = sum(
        1 for row in train_data
        if row[column] is None or row[column].strip() == ""
    )

    print(f"{column}: {missing} missing values")

    instructions = [row["instruction"] for row in train_data]

print("\nTotal instructions:", len(instructions))
print("Unique instructions:", len(set(instructions)))
print("Duplicate instructions:", len(instructions) - len(set(instructions)))


from collections import Counter

instruction_counts = Counter(instructions)

print("\n========== DUPLICATE EXAMPLES ==========")

shown = 0

for instruction, count in instruction_counts.items():
    if count > 1:
        print(f"\nInstruction: {instruction}")
        print(f"Occurrences: {count}")

        if shown == 4:
            break

        shown += 1



        print("\n========== CONFLICTING DUPLICATES ==========")

instruction_labels = {}

for row in train_data:
    instruction = row["instruction"]
    label = row["intent"]

    if instruction not in instruction_labels:
        instruction_labels[instruction] = set()

    instruction_labels[instruction].add(label)

conflicts = {
    instruction: labels
    for instruction, labels in instruction_labels.items()
    if len(labels) > 1
}

print("Number of conflicting instructions:", len(conflicts))

for instruction, labels in list(conflicts.items())[:10]:
    print("\nInstruction:", instruction)
    print("Different intents:", labels)