import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import os

base_dir = "../sample2/a-comprehensive-dataset-of-pattern-electroretinograms-for-ocular-electrophysiology-research-the-perg-ioba-dataset-1.0.0"
csv_dir = os.path.join(base_dir, "csv")
info_file = os.path.join(csv_dir,"participants_info.csv")

info_df = pd.read_csv(info_file)

print(info_df.columns)


info_df["label"] = info_df["diagnosis1"].apply(lambda x: 0 if str(x).strip().lower()== "normal" else 1)

features = []
labels = []

for _, row in info_df.iterrows():
    pid = str(row["id_record"]).zfill(4)
    signal_file = os.path.join(csv_dir, f"{pid}.csv")

    if os.path.exists(signal_file):
        try:
            df = pd.read_csv(signal_file)
            re = df["RE_1"]
            le = df["LE_1"]


            feature_vector =  [
                re.mean(), re.std(), re.min(), re.max(),re.skew(), re.kurt(),
                le.mean(), le.std(), le.min(), le.max(), le.skew(), le.kurt()
            ]

            features.append(feature_vector)
            labels.append(row["label"])
        except Exception as e:
            print(f"Skipping {pid}.csv due to error: {e}")

print(f"Number of valid samples: {len(features)}")
print(f"Number of labels: {len(labels)}")
print(f"Example features: {features[:1]}")
print("Unique values in diagnosis1:")
print(info_df["diagnosis1"].unique())

print("Number of NaN labels:")
print(info_df["label"].isna().sum())

X = np.array(features)
y = np.array(labels)

mask_valid = ~np.isnan(y)
X = X[mask_valid]
y = y[mask_valid]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state =42)

model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mask_pred_valid = ~np.isnan(y_test) & ~np.isnan(y_pred)
y_test_clean = y_test[mask_pred_valid]
y_pred_clean = y_pred[mask_pred_valid]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nclassification Report:")
print(classification_report(y_test,y_pred))

import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Abnormal (1)'],
            yticklabels=['Normal (0)', 'Abnormal (1)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')






