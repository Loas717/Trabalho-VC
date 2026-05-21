import shutil
from pathlib import Path

from ultralytics import YOLO


DATASET = Path("dataset_parking_cls")
MODELO_BASE = "yolo11n-cls.pt"
DESTINO_MODELO = Path("models/parking_occupancy_yolo11n_cls.pt")


def main():
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset preparado nao encontrado: {DATASET}. Rode primeiro: python preparar_dataset_classificacao.py"
        )

    model = YOLO(MODELO_BASE)
    resultados = model.train(
        data=str(DATASET),
        epochs=30,
        imgsz=128,
        batch=32,
        project="runs/classify",
        name="parking_occupancy_yolo11n",
        exist_ok=True,
    )

    melhor_modelo = Path(resultados.save_dir) / "weights" / "best.pt"
    if not melhor_modelo.exists():
        raise FileNotFoundError(f"Modelo treinado nao encontrado: {melhor_modelo}")

    DESTINO_MODELO.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(melhor_modelo, DESTINO_MODELO)
    print(f"\nModelo salvo em: {DESTINO_MODELO}")


if __name__ == "__main__":
    main()
