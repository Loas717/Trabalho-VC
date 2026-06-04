import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from cenarios import escolher_cenario, listar_cenarios


def parse_args():
    parser = argparse.ArgumentParser(description="Alinha uma imagem base externa com o video do cenario.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--base-externa", required=True, help="Imagem limpa a ser alinhada ao video.")
    parser.add_argument("--saida", help="Caminho da imagem alinhada. Padrao: cenarios/<cenario>/base_alinhada.png.")
    parser.add_argument("--frame-referencia", type=int, default=0, help="Frame do video usado como referencia.")
    parser.add_argument("--features", type=int, default=5000, help="Quantidade maxima de pontos ORB.")
    parser.add_argument("--aplicar", action="store_true", help="Substitui base.png pela imagem alinhada, criando backup.")
    return parser.parse_args()


def obter_cenario(nome):
    if not nome:
        return escolher_cenario("alinhar uma imagem base externa")

    for cenario in listar_cenarios():
        if cenario.nome == nome:
            print(f"Cenario selecionado: {cenario.nome}\n")
            return cenario

    raise FileNotFoundError(f"Cenario '{nome}' nao encontrado em cenarios/.")


def ler_frame_referencia(video_path, frame_referencia):
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise FileNotFoundError(f"Nao foi possivel abrir o video: {video_path}")

    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_referencia))
        check, frame = video.read()
    finally:
        video.release()

    if not check:
        raise RuntimeError(f"Nao foi possivel ler o frame {frame_referencia} do video.")

    return frame


def alinhar_por_homografia(base_externa, frame_referencia, features):
    base_cinza = cv2.cvtColor(base_externa, cv2.COLOR_BGR2GRAY)
    frame_cinza = cv2.cvtColor(frame_referencia, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max(500, features))
    kp_base, desc_base = orb.detectAndCompute(base_cinza, None)
    kp_frame, desc_frame = orb.detectAndCompute(frame_cinza, None)

    if desc_base is None or desc_frame is None:
        raise RuntimeError("Nao foram encontrados pontos suficientes para alinhar a imagem.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pares = matcher.knnMatch(desc_base, desc_frame, k=2)

    bons_matches = []
    for par in pares:
        if len(par) != 2:
            continue
        melhor, segundo = par
        if melhor.distance < 0.75 * segundo.distance:
            bons_matches.append(melhor)

    if len(bons_matches) < 12:
        raise RuntimeError(
            f"Poucos pontos casados para alinhar a base externa: {len(bons_matches)}. "
            "Tente uma imagem mais parecida com o frame do video."
        )

    pontos_base = np.float32([kp_base[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
    pontos_frame = np.float32([kp_frame[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
    homografia, mascara = cv2.findHomography(pontos_base, pontos_frame, cv2.RANSAC, 5.0)

    if homografia is None or mascara is None:
        raise RuntimeError("Nao foi possivel estimar a homografia entre a base externa e o video.")

    inliers = int(mascara.ravel().sum())
    if inliers < 10:
        raise RuntimeError(f"Alinhamento instavel: apenas {inliers} pontos consistentes.")

    altura, largura = frame_referencia.shape[:2]
    alinhada = cv2.warpPerspective(base_externa, homografia, (largura, altura))
    return alinhada, len(bons_matches), inliers


def aplicar_base(cenario, imagem_alinhada_path):
    destino = cenario.imagem_base
    backup = destino.with_name("base_backup.png")

    if destino.exists():
        shutil.copy2(destino, backup)
        print(f"Backup criado: {backup}")

    shutil.copy2(imagem_alinhada_path, destino)
    print(f"base.png atualizada: {destino}")


def main():
    args = parse_args()
    cenario = obter_cenario(args.cenario)
    base_path = Path(args.base_externa)
    saida = Path(args.saida) if args.saida else cenario.pasta / "base_alinhada.png"

    if not cenario.video.exists():
        raise FileNotFoundError(f"Video do cenario nao encontrado: {cenario.video}")
    if not base_path.exists():
        raise FileNotFoundError(f"Imagem base externa nao encontrada: {base_path}")

    base_externa = cv2.imread(str(base_path))
    if base_externa is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem base externa: {base_path}")

    frame = ler_frame_referencia(cenario.video, args.frame_referencia)
    alinhada, matches, inliers = alinhar_por_homografia(base_externa, frame, args.features)

    saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saida), alinhada)
    print(f"Imagem alinhada salva em: {saida}")
    print(f"Pontos casados: {matches}; pontos consistentes: {inliers}")

    if args.aplicar:
        aplicar_base(cenario, saida)


if __name__ == "__main__":
    main()
