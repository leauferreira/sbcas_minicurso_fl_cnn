from collections import OrderedDict
from pathlib import Path
from typing import List
import csv
import json

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from flwr.common import parameters_to_ndarrays
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =========================
# Configurações
# =========================
GLOBAL_TEST_DIR = Path("/app/global_test")
OUTPUT_DIR = Path("/app/output")
CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"
HISTORY_JSON = OUTPUT_DIR / "history.json"
HISTORY_CSV = OUTPUT_DIR / "history.csv"

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Histórico
# =========================
history_records = []
best_global_acc = -1.0


def save_history():
    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump(history_records, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "round",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "distributed_accuracy",
        "global_loss",
        "global_acc",
    ]

    with open(HISTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history_records:
            writer.writerow(row)


# =========================
# Dataset global de teste
# =========================
def build_global_test_loader():
    if not GLOBAL_TEST_DIR.exists():
        raise FileNotFoundError(f"Diretório global_test não encontrado: {GLOBAL_TEST_DIR}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(root=str(GLOBAL_TEST_DIR), transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"global_test carregado de: {GLOBAL_TEST_DIR}", flush=True)
    print(f"Classes encontradas: {dataset.classes}", flush=True)
    print(f"Total de amostras em global_test: {len(dataset)}", flush=True)

    return loader, len(dataset.classes)


GLOBAL_TEST_LOADER, NUM_CLASSES = build_global_test_loader()


# =========================
# Modelo global
# =========================
def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def save_model_checkpoint(parameters: List[np.ndarray], filepath: Path):
    model = build_model(NUM_CLASSES)
    set_parameters(model, parameters)
    torch.save(model.state_dict(), filepath)
    print(f"Checkpoint salvo em: {filepath}", flush=True)


@torch.no_grad()
def evaluate_global_model(model: nn.Module, loader: DataLoader):
    criterion = nn.CrossEntropyLoss()
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

    global_loss = running_loss / total
    global_acc = correct / total
    return global_loss, global_acc


# =========================
# Estratégia customizada
# =========================
class FedAvgWithGlobalTest(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None
        self.latest_fit_metrics = {}

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters

            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            round_ckpt = CHECKPOINTS_DIR / f"round_{server_round:03d}.pt"
            save_model_checkpoint(ndarrays, round_ckpt)

        self.latest_fit_metrics = aggregated_metrics or {}
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        global best_global_acc

        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        if aggregated_metrics is None:
            aggregated_metrics = {}

        global_loss = None
        global_acc = None

        if self.latest_parameters is not None:
            try:
                global_ndarrays = parameters_to_ndarrays(self.latest_parameters)
                model = build_model(NUM_CLASSES).to(DEVICE)
                set_parameters(model, global_ndarrays)

                global_loss, global_acc = evaluate_global_model(model, GLOBAL_TEST_LOADER)

                print(f"\n [ROUND {server_round}] Avaliação no global_test:", flush=True)
                print(f"   - global_loss: {global_loss:.4f}", flush=True)
                print(f"   - global_acc : {global_acc:.4f}", flush=True)

                aggregated_metrics["global_loss"] = float(global_loss)
                aggregated_metrics["global_acc"] = float(global_acc)

                if global_acc > best_global_acc:
                    best_global_acc = global_acc
                    best_ckpt = CHECKPOINTS_DIR / "best_model.pt"
                    save_model_checkpoint(global_ndarrays, best_ckpt)
                    print(f" Novo melhor modelo salvo (global_acc={global_acc:.4f})", flush=True)

            except Exception as exc:
                print(f"Falha ao avaliar no global_test: {exc}", flush=True)

        record = {
            "round": server_round,
            "train_loss": float(self.latest_fit_metrics.get("train_loss", np.nan)),
            "train_acc": float(self.latest_fit_metrics.get("train_acc", np.nan)),
            "val_loss": float(self.latest_fit_metrics.get("val_loss", np.nan)),
            "val_acc": float(self.latest_fit_metrics.get("val_acc", np.nan)),
            "distributed_accuracy": float(aggregated_metrics.get("accuracy", np.nan)),
            "global_loss": float(global_loss if global_loss is not None else np.nan),
            "global_acc": float(global_acc if global_acc is not None else np.nan),
        }
        history_records.append(record)
        save_history()

        print(f"\n Histórico da rodada {server_round}:", flush=True)
        for k, v in record.items():
            if k == "round":
                print(f"   - {k}: {v}", flush=True)
            else:
                print(f"   - {k}: {v:.4f}", flush=True)

        return aggregated_loss, aggregated_metrics


# =========================
# Agregação de métricas
# =========================
def weighted_average(metrics):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    aggregated = {}
    metric_names = set()

    for _, client_metrics in metrics:
        metric_names.update(client_metrics.keys())

    for metric_name in metric_names:
        weighted_sum = 0.0
        valid_examples = 0

        for num_examples, client_metrics in metrics:
            if metric_name in client_metrics:
                weighted_sum += num_examples * float(client_metrics[metric_name])
                valid_examples += num_examples

        if valid_examples > 0:
            aggregated[metric_name] = weighted_sum / valid_examples

    print("\n Métricas agregadas dos clientes:", flush=True)
    for key, value in aggregated.items():
        print(f"   - {key}: {value:.4f}", flush=True)

    return aggregated


# =========================
# Servidor Flower
# =========================
print("Servidor FL iniciado na porta 8080", flush=True)
print("USANDO ESTRATÉGIA: min_fit=3, min_eval=3, min_available=3", flush=True)

strategy = FedAvgWithGlobalTest(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=10),
    strategy=strategy,
)