import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

from gerar_base import gerar_base_mediana
from visao_vagas import CONFIG_PADRAO


DATASET_PADRAO = Path("datasets/parking-lot-detection-counter/parking")
CENARIO_PADRAO = Path("cenarios/kaggle_parking")


def parse_args():
    parser = argparse.ArgumentParser(description="Importa o dataset Parking Lot Detection Counter do Kaggle.")
    parser.add_argument("--dataset-dir", default=str(DATASET_PADRAO), help="Pasta extraida do dataset, contendo a pasta parking.")
    parser.add_argument("--cenario-dir", default=str(CENARIO_PADRAO), help="Pasta do cenario a criar/atualizar.")
    parser.add_argument("--amostras-base", type=int, default=120, help="Quantidade de frames usados para gerar base.png por mediana.")
    parser.add_argument("--area-minima", type=int, default=1000, help="Area minima de um componente da mascara para virar vaga.")
    parser.add_argument("--area-maxima", type=int, default=14000, help="Area maxima de um componente da mascara para virar vaga.")
    parser.add_argument("--sobrescrever", action="store_true", help="Sobrescreve arquivos existentes do cenario.")
    return parser.parse_args()


def resolver_dataset_dir(caminho):
    pasta = Path(caminho)
    if (pasta / "parking_1920_1080.mp4").exists():
        return pasta
    if (pasta / "parking" / "parking_1920_1080.mp4").exists():
        return pasta / "parking"
    raise FileNotFoundError(
        f"Dataset nao encontrado em {pasta}. Esperado: parking_1920_1080.mp4 e mask_1920_1080.png."
    )


def copiar_arquivo(origem, destino, sobrescrever):
    if destino.exists() and not sobrescrever:
        print(f"Mantendo arquivo existente: {destino}")
        return

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origem, destino)
    print(f"Copiado: {origem} -> {destino}")


def ordenar_pontos(pontos):
    pontos = np.array(pontos, dtype=np.float32)
    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1).reshape(-1)

    ordenados = np.zeros((4, 2), dtype=np.float32)
    ordenados[0] = pontos[np.argmin(soma)]
    ordenados[2] = pontos[np.argmax(soma)]
    ordenados[1] = pontos[np.argmin(diferenca)]
    ordenados[3] = pontos[np.argmax(diferenca)]
    return [(int(round(x)), int(round(y))) for x, y in ordenados]


def extrair_vagas_da_mascara(mask_path, area_minima, area_maxima):
    mascara = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a mascara: {mask_path}")

    _, binaria = cv2.threshold(mascara, 127, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vagas = []
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area < area_minima or area > area_maxima:
            continue

        retangulo = cv2.minAreaRect(contorno)
        largura, altura = retangulo[1]
        if largura <= 0 or altura <= 0:
            continue

        proporcao = max(largura, altura) / min(largura, altura)
        if proporcao < 1.2 or proporcao > 5.5:
            continue

        pontos = cv2.boxPoints(retangulo)
        vagas.append(ordenar_pontos(pontos))

    vagas.sort(key=lambda vaga: (min(y for _, y in vaga), min(x for x, _ in vaga)))
    return vagas


def salvar_config_kaggle(cenario_dir, sobrescrever):
    destino = cenario_dir / "config.json"
    if destino.exists() and not sobrescrever:
        print(f"Mantendo config existente: {destino}")
        return

    config = CONFIG_PADRAO.copy()
    config.update(
        {
            "limiar_pixels_diferentes": 24,
            "limiar_ocupacao": 0.16,
            "limiar_livre": 0.08,
            "limiar_diferenca_media": 0.030,
            "limiar_diferenca_media_livre": 0.018,
            "tamanho_recorte_vaga": 96,
            "analisar_a_cada_n_frames": 2,
            "mostrar_labels_vagas": False,
            "espessura_poligono": 1,
        }
    )
    with destino.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    print(f"Config criada: {destino}")


def main():
    args = parse_args()
    dataset_dir = resolver_dataset_dir(args.dataset_dir)
    cenario_dir = Path(args.cenario_dir)
    video_origem = dataset_dir / "parking_1920_1080.mp4"
    mascara_origem = dataset_dir / "mask_1920_1080.png"

    cenario_dir.mkdir(parents=True, exist_ok=True)
    video_destino = cenario_dir / "video.mp4"
    mascara_destino = cenario_dir / "mask.png"
    base_destino = cenario_dir / "base.png"
    coordenadas_destino = cenario_dir / "vagas_coordenadas.pkl"

    copiar_arquivo(video_origem, video_destino, args.sobrescrever)
    copiar_arquivo(mascara_origem, mascara_destino, args.sobrescrever)

    if not base_destino.exists() or args.sobrescrever:
        print("Gerando base.png por mediana de frames do video Kaggle...")
        base = gerar_base_mediana(video_destino, args.amostras_base, 0, 0)
        cv2.imwrite(str(base_destino), base)
        print(f"Base criada: {base_destino}")
    else:
        print(f"Mantendo base existente: {base_destino}")

    vagas = extrair_vagas_da_mascara(mascara_destino, args.area_minima, args.area_maxima)
    if not vagas:
        raise RuntimeError("Nenhuma vaga foi extraida da mascara. Ajuste area-minima/area-maxima.")

    if coordenadas_destino.exists() and not args.sobrescrever:
        print(f"Mantendo coordenadas existentes: {coordenadas_destino}")
    else:
        with coordenadas_destino.open("wb") as f:
            pickle.dump(vagas, f)
        print(f"{len(vagas)} vaga(s) salvas em: {coordenadas_destino}")

    salvar_config_kaggle(cenario_dir, args.sobrescrever)
    print("\nCenario pronto. Rode:")
    print("python main.py --cenario kaggle_parking")


if __name__ == "__main__":
    main()
