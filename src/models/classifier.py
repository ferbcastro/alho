import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import sys

if len(sys.argv) < 2:
    print("No dataset specified")
    exit(1)

# Load Data

df = pd.read_csv(sys.argv[1])

X = df.drop(["label", "url"], axis=1).values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train

model = lgb.LGBMClassifier(
    objective="binary",
    learning_rate=0.1,
    num_leaves=64,
    device_type="gpu",
    n_estimators=500,
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

