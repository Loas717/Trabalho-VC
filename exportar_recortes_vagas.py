import argparse
import pickle
from pathlib import Path

import cv2

from cenarios import escolher_cenario, listar_cenarios
from visao_vagas import carregar_config_cenario, recortar_vaga_normalizada


def parse_args():
    parser = argparse.ArgumentParser(description="Exporta recortes normalizados das vagas para rotulagem.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--config", help="Caminho para um JSON de configuracao alternativo.")
    parser.add_argument("--saida", default="dataset_recortes", help="Pasta onde os recortes serao salvos.")
    parser.add_argument("--intervalo", type=int, default=30, help="Exporta um frame a cada N frames.")
    parser.add_argument("--inicio", type=int, default=1, help="Primeiro frame considerado.")
    parser.add_argument("--limite", type=int, default=0, help="Numero maximo de frames lidos. Use 0 para processar tudo.")
    return parser.parse_args()


def obter_cenario(nome):
    if not nome:
        return escolher_cenario("exportar recortes das vagas")

    for cenario in listar_cenarios():
        if cenario.nome == nome:
            print(f"Cenario selecionado: {cenario.nome}\n")
            return cenario

    raise FileNotFoundError(f"Cenario '{nome}' nao encontrado em cenarios/.")


def main():
    args = parse_args()
    cenario = obter_cenario(args.cenario)
    config = carregar_config_cenario(cenario, args.config)

    if not cenario.video.exists():
        raise FileNotFoundError(f"Video do cenario nao encontrado: {cenario.video}")

    if not cenario.coordenadas.exists():
        raise FileNotFoundError(
            f"Coordenadas nao encontradas: {cenario.coordenadas}. Rode primeiro: python selector.py"
        )

    with cenario.coordenadas.open("rb") as f:
        vagas = pickle.load(f)

    saida = Path(args.saida) / cenario.nome / "sem_rotulo"
    saida.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(cenario.video))
    numero_frame = 0
    total_recortes = 0
    tamanho_recorte = int(config["tamanho_recorte_vaga"])

    try:
        while True:
            check, frame = video.read()
            if not check:
                break

            numero_frame += 1
            if args.limite and numero_frame > args.limite:
                break

            if numero_frame < args.inicio:
                continue

            if (numero_frame - args.inicio) % max(1, args.intervalo) != 0:
                continue

            for idx, vaga in enumerate(vagas, start=1):
                recorte = recortar_vaga_normalizada(frame, vaga, tamanho=tamanho_recorte)
                if recorte is None:
                    continue

                nome = f"{cenario.nome}_frame_{numero_frame:06d}_vaga_{idx:03d}.jpg"
                cv2.imwrite(str(saida / nome), recorte)
                total_recortes += 1
    finally:
        video.release()

    print(f"Recortes salvos em: {saida}")
    print(f"Total de recortes exportados: {total_recortes}")
    print("Revise as imagens e mova manualmente para dataset/empty e dataset/occupied antes de treinar.")


if __name__ == "__main__":
    main()
