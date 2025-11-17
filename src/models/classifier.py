import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import sys

bigrams_train = "dataset/bigrams/train_set_bigrams.csv"
bigrams_test = "dataset/bigrams/test_set_bigrams.csv"
tetragrams_train = "dataset/tetragrams/train_4_extract.csv"
tetragrams_test = "dataset/tetragrams/test_4_extract.csv"

# Load Data

print("Loading files")

btrain = pd.read_csv(bigrams_train)
btest = pd.read_csv(bigrams_test)
ttrain = pd.read_csv(tetragrams_train).drop(["label", "url"], axis=1)
ttest = pd.read_csv(tetragrams_test).drop(["label", "url"], axis=1)

train = pd.concat([btrain, ttrain], axis=1)
test = pd.concat([btest, ttest], axis=1)

print("Preparing datasets")

X_train = train.drop(["label", "url"], axis=1).values
y_train = train["label"].values

X_test = test.drop(["label", "url"], axis=1).values
y_test = test["label"].values

"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)
"""

# Train

print("Training...")

model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.1,
    num_leaves=32,
    device_type="gpu",
    n_estimators=100,
)

model.fit(X_train, y_train)

# Test
prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

# Evaluate

precision = precision_score(y_test, pred, average="binary")
recall = recall_score(y_test, pred, average="binary")
f1 = f1_score(y_test, pred, average="binary")
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
print("Recall:", recall)
print("F1-Score:", f1)
print("Precision:", precision)

