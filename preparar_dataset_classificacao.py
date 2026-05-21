import random
import shutil
from pathlib import Path


DATASET_ORIGEM = Path("dataset")
DATASET_DESTINO = Path("dataset_parking_cls")
CLASSES = ["empty", "occupied"]
VALIDACAO = 0.2
SEMENTE = 42


def limpar_destino():
    if DATASET_DESTINO.exists():
        shutil.rmtree(DATASET_DESTINO)


def copiar_split(classe, arquivos):
    random.shuffle(arquivos)
    total_validacao = max(1, int(len(arquivos) * VALIDACAO))
    splits = {
        "val": arquivos[:total_validacao],
        "train": arquivos[total_validacao:],
    }

    for split, split_arquivos in splits.items():
        destino = DATASET_DESTINO / split / classe
        destino.mkdir(parents=True, exist_ok=True)
        for arquivo in split_arquivos:
            shutil.copy2(arquivo, destino / arquivo.name)

        print(f"{split}/{classe}: {len(split_arquivos)} imagens")


def main():
    random.seed(SEMENTE)
    limpar_destino()

    for classe in CLASSES:
        pasta = DATASET_ORIGEM / classe
        if not pasta.exists():
            raise FileNotFoundError(f"Pasta nao encontrada: {pasta}")

        arquivos = sorted(
            list(pasta.glob("*.jpg"))
            + list(pasta.glob("*.jpeg"))
            + list(pasta.glob("*.png"))
        )
        if not arquivos:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {pasta}")

        copiar_split(classe, arquivos)

    print(f"\nDataset preparado em: {DATASET_DESTINO}")


if __name__ == "__main__":
    main()
