import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
import torchaudio
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#putanja do obradenih podataka
DATA_DIR = "C:/Fakultet Programs/Zavrsni Rad Kod/project/data_processed_model2"

print("Učitavanje...")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Koristim uređaj:", device)

#dohvacanje kategorija
categories = []
for d in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, d)):
        categories.append(d)

categories = sorted(categories)
print("Pronađene kategorije u folderu:", categories)

id2label = {}
label2id = {}

for idx, name in enumerate(categories):
    id2label[idx] = name
    label2id[name] = idx

#ucitavanje modela i konfiguracije
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"

feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)

config = ASTConfig.from_pretrained(
    model_name,
    num_labels=10,
    id2label=id2label,
    label2id=label2id
)

model = ASTForAudioClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

model.to(device)

#funkcija za AST znacajke
def get_ast_features(file_path):
    waveform, sr = torchaudio.load(file_path)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze().numpy()
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    return inputs.input_values.to(device)

#priprema podataka
X_paths = []
y = []

for label_name in categories:
    label_id = label2id[label_name]
    folder_path = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            X_paths.append(full_path)
            y.append(label_id)

X_train_p, X_test_p, y_train, y_test = train_test_split(
    X_paths, y, test_size=0.2, random_state=42, stratify=y
)

#evaluacija
model.eval()
predictions = []

print("\nAST analizira testne podatke za 10 stvarnih klasa...")

with torch.no_grad():
    for path in X_test_p:
        input_values = get_ast_features(path)
        outputs = model(input_values)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred)

#rezultati
accuracy = accuracy_score(y_test, predictions)
print("\nUkupna točnost AST modela za 10 klasa:", round(accuracy, 2))

print("\nDetaljan izvještaj po klasama:")
print(classification_report(y_test, predictions, target_names=categories))

df = pd.DataFrame({"True": y_test, "Pred": predictions})
df.to_csv("ast_10_classes_results.csv", index=False)

#print("Rezultati spremljeni u ast_10_classes_results.csv")
