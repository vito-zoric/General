import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification, ASTConfig
import torchaudio
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

#putanja do obradenih podataka

DATA_DIR = "C:/Fakultet Programs/Zavrsni Rad Kod/project/data_processed_model1"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5

print("Učitavanje...")
print("WARNING: Testiranje može trajati do 1.5h")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Koristim uređaj:", device)

categories = []
for d in os.listdir(DATA_DIR):
    if os.path.isdir(os.path.join(DATA_DIR, d)):
        categories.append(d)

categories = sorted(categories)

id2label = {}
label2id = {}

for idx, name in enumerate(categories):
    id2label[idx] = name
    label2id[name] = idx

NUM_LABELS = len(categories)

model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)


class AudioDataset(Dataset):
    def __init__(self, paths, labels, extractor):
        self.paths = paths
        self.labels = labels
        self.extractor = extractor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        waveform, sr = torchaudio.load(path)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        waveform = waveform.squeeze().numpy()
        inputs = self.extractor(waveform, sampling_rate=16000, return_tensors="pt")

        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.long)


#priprema datoteka
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

train_dataset = AudioDataset(X_train_p, y_train, feature_extractor)
test_dataset = AudioDataset(X_test_p, y_test, feature_extractor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#inicijalizacija modela

config = ASTConfig.from_pretrained(model_name)
config.num_labels = NUM_LABELS
config.id2label = id2label
config.label2id = label2id

model = ASTForAudioClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True
)

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#petlja za treniranje


print("\nZapočinje Fine-tuning za", NUM_LABELS, "klase (Binarni model)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_inputs, batch_labels in tqdm(train_loader, desc="Epoha " + str(epoch + 1) + "/" + str(EPOCHS)):
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_inputs)
        loss = criterion(outputs.logits, batch_labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print("Epoha", epoch + 1, "završena. Prosječni Loss:", round(avg_loss, 4))

#evaluacija i ispis

model.eval()
predictions = []
true_labels = []

print("\nEvaluacija modela na testnim podacima...")

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(device)

        outputs = model(batch_inputs)
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        for p in preds:
            predictions.append(p)

        for t in batch_labels.numpy():
            true_labels.append(t)

accuracy = accuracy_score(true_labels, predictions)
print("\nUkupna točnost nakon fine-tuninga za binary:", round(accuracy, 2))

print("\nDetaljan izvještaj po klasama:")
print(classification_report(true_labels, predictions, target_names=categories))

torch.save(model.state_dict(), "ast_binary_finetuned.pt")
#print("Težinski faktori spremljeni u ast_binary_finetuned.pt")
