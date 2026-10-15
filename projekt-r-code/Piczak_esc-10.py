import os
import numpy as np
import pandas as pd
from features import AudioFeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#putanja do obrađenih podataka
DATA_DIR = "C:/Fakultet Programs/Zavrsni Rad Kod/project/data_processed_model2"
extractor = AudioFeatureExtractor()

print("Učitavanje...")

X = []
y = []

#dohvacanje svih 10 kategorija
categories = []
for d in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, d)):
        categories.append(d)

categories = sorted(categories)

category_to_id = {}
for idx, name in enumerate(categories):
    category_to_id[name] = idx

print("Kategorije koje model uči:", categories)
print("Započinjem ekstrakciju značajki za 10 klasa...")

#ekstrakcija znacajki
for label_name in categories:
    label_id = category_to_id[label_name]
    folder_path = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)
            features = extractor.extract(file_path)
            X.append(features)
            y.append(label_id)

X = np.array(X)
y = np.array(y)

#podjela na trening i test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#random forest model
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

#rezultati
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("\nUkupna točnost Piczak modela za 10 klasa:", round(accuracy, 2))

print("\nDetaljan izvještaj po klasama:")
print(classification_report(y_test, predictions, target_names=categories))

#spremanje rezultata
df = pd.DataFrame({
    "True": y_test,
    "Pred": predictions
})

df.to_csv("piczak_10_classes_results.csv", index=False)
