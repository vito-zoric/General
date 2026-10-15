import os
import numpy as np
from model1.features import AudioFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

#putanja do obradenih podataka
DATA_DIR = "C:/Fakultet Programs/Zavrsni Rad Kod/project/data_processed_model1"
extractor = AudioFeatureExtractor()

X = []
y = []

print("Učitavanje...")

for label_name in ["no_accident", "accident"]:
    if label_name == "accident":
        label_id = 1
    else:
        label_id = 0

    folder_path = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        features = extractor.extract(file_path)
        X.append(features)
        y.append(label_id)

X = np.array(X)
y = np.array(y)

#podjela na trening i test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#random forest model
model = RandomForestClassifier(n_estimators=500)
model.fit(X_train, y_train)

#evaluacija
predictions = model.predict(X_test)

print("Točnost modela:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

#spremanje rezultata
results = pd.DataFrame({
    "Stvarno": y_test,
    "Predviđeno": predictions
})

results.to_csv("piczak_results.csv", index=False)

