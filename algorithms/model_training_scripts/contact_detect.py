# pylint: disable=all

import os
import torch
from ultralytics import YOLO
import cv2
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import yaml

DATA_YAML_PATH = "/home/daniel/Documentos/projetos/Marmoset-Detector/dataset/yolov11_dataset/data.yaml"
BEST_MODEL_PATH = (
    "/home/daniel/Documentos/projetos/Marmoset-Detector/trained_data/weights/best.pt"
)
NEW_MODEL_PATH = (
    "/home/daniel/Documentos/projetos/Marmoset-Detector/trained_data/weights/new.pt"
)
BATCH_SIZE = 8
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_count = torch.cuda.device_count()
    print(f"GPU(s) detectada(s): {gpu_count}. Utilizando a GPU para treinamento.")
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")
    print("GPU não detectada. Utilizando a CPU para treinamento.")

print("Carregando o modelo YOLO para detecção de saguis...")
if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"O modelo YOLO especificado não foi encontrado: {BEST_MODEL_PATH}"
    )
yolo_model = YOLO(BEST_MODEL_PATH)
yolo_model.to(device)


def masks_in_contact(mask1, mask2, touch_threshold=1):
    intersection = cv2.bitwise_and(mask1, mask2)
    if cv2.countNonZero(intersection) > 0:
        return True

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (touch_threshold, touch_threshold)
    )
    dilated_mask1 = cv2.dilate(mask1, kernel, iterations=1)
    contact = cv2.bitwise_and(dilated_mask1, mask2)
    return cv2.countNonZero(contact) > 0


def generate_contact_labels(data_yaml_path, yolo_model, device):
    with open(data_yaml_path, "r") as file:
        data_yaml = yaml.safe_load(file)

    base_dir = os.path.dirname(os.path.abspath(data_yaml_path))

    train_dir_rel = data_yaml.get("train", "")
    val_dir_rel = data_yaml.get("val", "")
    test_dir_rel = data_yaml.get("test", "")

    train_dir = os.path.join(base_dir, train_dir_rel) if train_dir_rel else ""
    val_dir = os.path.join(base_dir, val_dir_rel) if val_dir_rel else ""
    test_dir = os.path.join(base_dir, test_dir_rel) if test_dir_rel else ""

    def process_directory(image_dir):
        data = []
        if not os.path.exists(image_dir):
            print(f"Diretório não encontrado: {image_dir}. Pulando...")
            return data

        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        for img_file in tqdm(image_files, desc=f"Processando {image_dir}"):
            img_path = os.path.join(image_dir, img_file)
            if not os.path.exists(img_path):
                print(f"Imagem não encontrada: {img_path}. Pulando...")
                continue
            results = yolo_model.predict(img_path, verbose=False)
            masks = []
            for result in results:
                if result.masks:
                    for mask in result.masks.data:
                        masks.append(mask.cpu().numpy())

            contact = 0
            for mask1, mask2 in combinations(masks, 2):
                if masks_in_contact(mask1, mask2):
                    contact = 1
                    break
            data.append({"image_path": img_path, "label": contact})
        return data

    train_data = process_directory(train_dir) if train_dir else []
    val_data = process_directory(val_dir) if val_dir else []
    test_data = process_directory(test_dir) if test_dir else []

    all_data = pd.DataFrame(train_data + val_data + test_data)
    return all_data


print("Gerando rótulos de contato para o dataset...")
dataset_df = generate_contact_labels(DATA_YAML_PATH, yolo_model, device)


class MarmosetContactDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label"]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Imagem não encontrada ou inválida: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

if not dataset_df.empty:
    if "label" not in dataset_df.columns or "image_path" not in dataset_df.columns:
        raise ValueError("O DataFrame deve conter as colunas 'image_path' e 'label'.")

    unique_labels = dataset_df["label"].nunique()
    if unique_labels < 2:
        print(
            "Aviso: O dataset possui menos de duas classes. A divisão pode não ser estratificada."
        )
        train_df, temp_df = train_test_split(dataset_df, test_size=0.3, random_state=42)
    else:
        train_df, temp_df = train_test_split(
            dataset_df, test_size=0.3, stratify=dataset_df["label"], random_state=42
        )

    if not temp_df.empty:
        if "label" in temp_df.columns and temp_df["label"].nunique() >= 2:
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
            )
        else:
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    else:
        val_df, test_df = pd.DataFrame(), pd.DataFrame()
else:
    train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

if train_df.empty and val_df.empty and test_df.empty:
    raise ValueError(
        "Nenhum dado encontrado após o processamento. Verifique as anotações e os caminhos no data.yaml."
    )

train_dataset = MarmosetContactDataset(train_df, transform=transform)
val_dataset = MarmosetContactDataset(val_df, transform=transform)
test_dataset = MarmosetContactDataset(test_df, transform=transform)

num_workers = 4 if os.name != "nt" else 0

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
)


class ContactClassifier(nn.Module):
    def __init__(self):
        super(ContactClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


model = ContactClassifier().to(device)

if torch.cuda.device_count() > 1:
    print(
        f"Múltiplas GPUs detectadas: {torch.cuda.device_count()}. Utilizando DataParallel."
    )
    model = nn.DataParallel(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def calculate_accuracy(y_pred, y_true):
    y_pred_label = (y_pred > 0.5).float()
    y_true_label = (y_true > 0.5).float()
    return (y_pred_label == y_true_label).float().mean()


print("Iniciando o treinamento do classificador de contato...")

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in tqdm(
        train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Treinamento"
    ):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_acc += acc.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_acc / len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} - Validação"
        ):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_acc += acc.item() * images.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Treino Loss: {epoch_loss:.4f}, Acurácia: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Acurácia: {val_acc:.4f}"
    )

print("Treinamento concluído.")

model.eval()
test_loss = 0.0
test_acc = 0.0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Avaliando no conjunto de teste"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = calculate_accuracy(outputs, labels)

        test_loss += loss.item() * images.size(0)
        test_acc += acc.item() * images.size(0)

test_loss /= len(test_loader.dataset)
test_acc /= len(test_loader.dataset)

print(f"Conjunto de Teste - Loss: {test_loss:.4f}, Acurácia: {test_acc:.4f}")

if isinstance(model, nn.DataParallel):
    torch.save(model.module.state_dict(), NEW_MODEL_PATH)
else:
    torch.save(model.state_dict(), NEW_MODEL_PATH)

print(f"Modelo treinado salvo em {NEW_MODEL_PATH}")
