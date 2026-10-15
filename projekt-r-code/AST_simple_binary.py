import torch
import torch.nn as nn
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torchaudio
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#putanja do obradenih podataka
DATA_DIR = "C:/Fakultet Programs/Zavrsni Rad Kod/project/data_processed_model1"

print("Učitavanje...")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Koristim uređaj:", device)

#ucitavanje modela i feature extractora
model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
model = ASTForAudioClassification.from_pretrained(model_name)

#prilagodba modela za 2 klase
model.classifier.dense = nn.Linear(model.config.hidden_size, 2)
model.num_labels = 2
model.to(device)


def get_ast_features(file_path):
    waveform, sr = torchaudio.load(file_path)

    #AST ocekuje 16 kHz pa resamplamo ako treba
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    waveform = waveform.squeeze().numpy()
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")

    return inputs.input_values.to(device)


#priprema podataka
X_paths = []
y = []

for label_name in ["no_accident", "accident"]:
    if label_name == "accident":
        label_id = 1
    else:
        label_id = 0

    folder_path = os.path.join(DATA_DIR, label_name)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            full_path = os.path.join(folder_path, file)
            X_paths.append(full_path)
            y.append(label_id)

X_train_p, X_test_p, y_train, y_test = train_test_split(
    X_paths, y, test_size=0.2, random_state=42
)

#jednostavno testiranje (inference)
model.eval()
predictions = []

print("AST analizira testne podatke...")

with torch.no_grad():
    for path in X_test_p:
        input_values = get_ast_features(path)
        outputs = model(input_values)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred)

#rezultati
accuracy = accuracy_score(y_test, predictions)
print("Točnost AST modela:", accuracy)
print(classification_report(y_test, predictions))

#spremanje rezultata
df = pd.DataFrame({"True": y_test, "Pred": predictions})
df.to_csv("ast_results.csv", index=False)
