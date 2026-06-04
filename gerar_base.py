import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from cenarios import escolher_cenario, listar_cenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Gera uma imagem base do estacionamento por mediana de frames.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--amostras", type=int, default=90, help="Quantidade de frames usados na mediana.")
    parser.add_argument("--inicio", type=int, default=0, help="Primeiro frame elegivel.")
    parser.add_argument("--fim", type=int, default=0, help="Ultimo frame elegivel. Use 0 para ir ate o fim.")
    parser.add_argument("--saida", help="Caminho da imagem gerada. Padrao: cenarios/<cenario>/base_gerada.png.")
    parser.add_argument("--aplicar", action="store_true", help="Substitui base.png pela imagem gerada, criando backup.")
    return parser.parse_args()


def obter_cenario(nome):
    if not nome:
        return escolher_cenario("gerar a imagem base")

    for cenario in listar_cenarios():
        if cenario.nome == nome:
            print(f"Cenario selecionado: {cenario.nome}\n")
            return cenario

    raise FileNotFoundError(f"Cenario '{nome}' nao encontrado em cenarios/.")


def posicoes_amostragem(total_frames, inicio, fim, quantidade):
    inicio = max(0, inicio)
    fim = total_frames - 1 if fim <= 0 else min(fim, total_frames - 1)
    if fim < inicio:
        raise ValueError("Intervalo invalido: fim menor que inicio.")

    quantidade = max(1, min(quantidade, fim - inicio + 1))
    return np.linspace(inicio, fim, quantidade, dtype=np.int32)


def ler_frames_amostrados(video, posicoes):
    frames = []

    for posicao in posicoes:
        video.set(cv2.CAP_PROP_POS_FRAMES, int(posicao))
        check, frame = video.read()
        if not check:
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError("Nenhum frame foi lido do video.")

    return frames


def gerar_base_mediana(video_path, amostras, inicio, fim):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {video_path}")

    try:
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError("Nao foi possivel descobrir a quantidade de frames do video.")

        posicoes = posicoes_amostragem(total_frames, inicio, fim, amostras)
        print(f"Lendo {len(posicoes)} frame(s) entre {posicoes[0]} e {posicoes[-1]}...")
        frames = ler_frames_amostrados(video, posicoes)
    finally:
        video.release()

    pilha = np.stack(frames, axis=0)
    return np.median(pilha, axis=0).astype(np.uint8)


def aplicar_base(cenario, imagem_gerada_path):
    destino = cenario.imagem_base
    backup = destino.with_name("base_backup.png")

    if destino.exists():
        shutil.copy2(destino, backup)
        print(f"Backup criado: {backup}")

    shutil.copy2(imagem_gerada_path, destino)
    print(f"base.png atualizada: {destino}")


def main():
    args = parse_args()
    cenario = obter_cenario(args.cenario)

    if not cenario.video.exists():
        raise FileNotFoundError(f"Video do cenario nao encontrado: {cenario.video}")

    saida = Path(args.saida) if args.saida else cenario.pasta / "base_gerada.png"
    saida.parent.mkdir(parents=True, exist_ok=True)

    base = gerar_base_mediana(cenario.video, args.amostras, args.inicio, args.fim)
    cv2.imwrite(str(saida), base)
    print(f"Imagem base gerada em: {saida}")

    if args.aplicar:
        aplicar_base(cenario, saida)


if __name__ == "__main__":
    main()
