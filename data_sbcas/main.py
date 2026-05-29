import random
import shutil
from pathlib import Path

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# =========================
# Configurações fixas
# =========================
SEED = 42
NUM_CLIENTS = 3
COPY_FILES = True  # True = copia arquivos | False = cria links simbólicos

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "SingleCellPAP"
TRAIN_DIR = DATASET_DIR / "Training"
TEST_DIR = DATASET_DIR / "Test"
OUTPUT_DIR = BASE_DIR / "data"


def list_class_dirs(root_dir: Path):
    return sorted([p for p in root_dir.iterdir() if p.is_dir()], key=lambda x: x.name)


def list_images(class_dir: Path):
    return sorted(
        [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ],
        key=lambda x: x.name,
    )


def safe_remove_dir(dir_path: Path):
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)


def safe_link_or_copy(src: Path, dst: Path, copy_files: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def split_evenly(items, num_parts):
    parts = [[] for _ in range(num_parts)]
    for idx, item in enumerate(items):
        parts[idx % num_parts].append(item)
    return parts


def validate_structure():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {DATASET_DIR}")

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {TRAIN_DIR}")

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {TEST_DIR}")

    train_classes = [p.name for p in list_class_dirs(TRAIN_DIR)]
    test_classes = [p.name for p in list_class_dirs(TEST_DIR)]

    if not train_classes:
        raise ValueError("Nenhuma classe encontrada em Training.")

    if not test_classes:
        raise ValueError("Nenhuma classe encontrada em Test.")

    if sorted(train_classes) != sorted(test_classes):
        raise ValueError(
            "As classes em Training e Test não coincidem.\n"
            f"Training: {train_classes}\n"
            f"Test: {test_classes}"
        )

    return sorted(train_classes)


def prepare_output_dirs(class_names):
    safe_remove_dir(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_CLIENTS + 1):
        client_dir = OUTPUT_DIR / f"client_{i}"
        client_dir.mkdir(parents=True, exist_ok=True)
        for cls in class_names:
            (client_dir / cls).mkdir(parents=True, exist_ok=True)

    global_test_dir = OUTPUT_DIR / "global_test"
    global_test_dir.mkdir(parents=True, exist_ok=True)
    for cls in class_names:
        (global_test_dir / cls).mkdir(parents=True, exist_ok=True)


def copy_global_test(class_names):
    summary = {}

    for cls in class_names:
        class_dir = TEST_DIR / cls
        images = list_images(class_dir)

        if not images:
            raise ValueError(f"Nenhuma imagem encontrada em {class_dir}")

        summary[cls] = len(images)

        for img_path in images:
            dst = OUTPUT_DIR / "global_test" / cls / img_path.name
            safe_link_or_copy(img_path, dst, COPY_FILES)

    return summary


def split_training_among_clients(class_names):
    random.seed(SEED)
    summary = {f"client_{i}": {cls: 0 for cls in class_names} for i in range(1, NUM_CLIENTS + 1)}

    for cls in class_names:
        class_dir = TRAIN_DIR / cls
        images = list_images(class_dir)

        if not images:
            raise ValueError(f"Nenhuma imagem encontrada em {class_dir}")

        random.shuffle(images)
        client_splits = split_evenly(images, NUM_CLIENTS)

        for client_idx, split_imgs in enumerate(client_splits, start=1):
            client_name = f"client_{client_idx}"

            for img_path in split_imgs:
                dst = OUTPUT_DIR / client_name / cls / img_path.name
                safe_link_or_copy(img_path, dst, COPY_FILES)
                summary[client_name][cls] += 1

    return summary


def print_summary(train_summary, test_summary):
    print("\n=== Estrutura gerada com sucesso ===")
    print(f"Dataset base : {DATASET_DIR}")
    print(f"Training     : {TRAIN_DIR}")
    print(f"Test         : {TEST_DIR}")
    print(f"Saída        : {OUTPUT_DIR}")
    print(f"Número de clientes: {NUM_CLIENTS}")
    print(f"Modo         : {'cópia' if COPY_FILES else 'link simbólico'}")

    print("\n=== global_test ===")
    total_test = sum(test_summary.values())
    print(f"Total: {total_test}")
    for cls, count in test_summary.items():
        print(f"  {cls}: {count}")

    print("\n=== Resumo por cliente (Training dividido aleatoriamente) ===")
    for client_name, class_counts in train_summary.items():
        total = sum(class_counts.values())
        print(f"\n{client_name} - total: {total}")
        for cls, count in class_counts.items():
            print(f"  {cls}: {count}")


def main():
    class_names = validate_structure()
    prepare_output_dirs(class_names)

    test_summary = copy_global_test(class_names)
    train_summary = split_training_among_clients(class_names)

    print_summary(train_summary, test_summary)


if __name__ == "__main__":
    main()