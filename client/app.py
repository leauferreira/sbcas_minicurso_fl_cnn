import os
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


# =========================
# Configurações
# =========================
DATA_DIR = Path("/app/data")
SERVER_ADDRESS = "fl-server:8080"

BATCH_SIZE = 16
LOCAL_EPOCHS = 3
LEARNING_RATE = 1e-3
VAL_RATIO = 0.2
RANDOM_SEED = 42
NUM_RETRIES = 20
RETRY_SLEEP = 3
INITIAL_SLEEP = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


# =========================
# Inspeção do volume montado
# =========================
def inspect_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        raise FileNotFoundError(f"Pasta de dados não encontrada: {data_dir}")

    class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
    if not class_dirs:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em {data_dir}")

    total_files = 0
    log(f"📁 Pasta de dados encontrada: {data_dir}")
    log("📚 Classes e quantidades:")

    for class_dir in class_dirs:
        num_files = len([p for p in class_dir.iterdir() if p.is_file()])
        total_files += num_files
        log(f"   - {class_dir.name}: {num_files} arquivos")

    log(f"🖼️ Total de arquivos em /app/data: {total_files}")


# =========================
# Dataset e DataLoader
# =========================
def get_dataloaders():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    base_dataset = datasets.ImageFolder(root=str(DATA_DIR))
    class_names = base_dataset.classes
    num_classes = len(class_names)

    dataset_size = len(base_dataset)
    val_size = max(1, int(dataset_size * VAL_RATIO))
    train_size = dataset_size - val_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # Wrappers simples para aplicar transforms diferentes
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            image, label = self.subset[idx]
            image = image.convert("RGB")
            image = self.transform(image)
            return image, label

    train_dataset = TransformedSubset(train_subset, train_transform)
    val_dataset = TransformedSubset(val_subset, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    log(f"Dataset carregado com sucesso")
    log(f"Classes encontradas: {class_names}")
    log(f"Número de classes: {num_classes}")
    log(f"️Amostras de treino: {len(train_dataset)}")
    log(f"Amostras de validação: {len(val_dataset)}")

    return train_loader, val_loader, num_classes


# =========================
# Modelo
# =========================
def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Congela backbone
    for param in model.parameters():
        param.requires_grad = False

    # Substitui cabeça
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================
# Funções de treino/avaliação
# =========================
def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    return loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total
    acc = correct / total
    return loss, acc


# =========================
# Conversão de parâmetros
# =========================
def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=True)


# =========================
# Inicialização
# =========================
log("Cliente iniciado")
log(f"Dispositivo: {DEVICE}")

inspect_data_dir(DATA_DIR)
train_loader, val_loader, num_classes = get_dataloaders()

model = build_model(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)


# =========================
# Cliente Flower
# =========================
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        log("get_parameters chamado")
        return get_parameters(model)

    def fit(self, parameters, config):
        log("Recebendo parâmetros globais")
        set_parameters(model, parameters)

        for epoch in range(LOCAL_EPOCHS):
            train_loss, train_acc = train(model, train_loader, criterion, optimizer)
            log(
                f"Epoch {epoch + 1}/{LOCAL_EPOCHS} "
                f"- train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )

        val_loss, val_acc = evaluate(model, val_loader, criterion)
        log(f"Fim treino local - val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        return get_parameters(model), len(train_loader.dataset), {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }

    def evaluate(self, parameters, config):
        log("Avaliando modelo global")
        set_parameters(model, parameters)

        loss, acc = evaluate(model, val_loader, criterion)
        log(f"evaluate -> loss={loss:.4f} acc={acc:.4f}")

        return float(loss), len(val_loader.dataset), {"accuracy": float(acc)}


# =========================
# Conexão com servidor
# =========================
time.sleep(INITIAL_SLEEP)

for i in range(NUM_RETRIES):
    try:
        log(f"Tentando conectar ao servidor (tentativa {i + 1})...")
        fl.client.start_client(
            server_address=SERVER_ADDRESS,
            client=FlowerClient().to_client(),
        )
        break
    except Exception as e:
        log(f"Falha ao conectar: {e}")
        time.sleep(RETRY_SLEEP)