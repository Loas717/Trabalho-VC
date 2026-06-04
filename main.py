import argparse
import csv
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from cenarios import escolher_cenario, listar_cenarios
from visao_vagas import (
    analisar_vagas_por_subtracao,
    carregar_config_cenario,
    preparar_bases_vagas,
    salvar_config_exemplo,
)


TITULO_JANELA = "Deteccao por subtracao de fundo"
LARGURA_JANELA = 1280
ALTURA_JANELA = 720


def parse_args():
    parser = argparse.ArgumentParser(description="Detecta vagas livres e ocupadas por subtracao de fundo.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--config", help="Caminho para um config.json alternativo.")
    parser.add_argument("--base", help="Imagem base alternativa, util para testar uma base sem carros.")
    parser.add_argument("--salvar-debug", action="store_true", help="Salva CSV, frames anotados e recortes analisados.")
    parser.add_argument("--debug-dir", default="debug", help="Pasta de saida do modo debug.")
    parser.add_argument("--debug-a-cada", type=int, default=30, help="Intervalo de frames para salvar imagens de debug.")
    parser.add_argument("--limite-frames", type=int, default=0, help="Para apos N frames. Use 0 para processar tudo.")
    parser.add_argument("--sem-janela", action="store_true", help="Processa sem abrir cv2.imshow.")
    return parser.parse_args()


def obter_cenario(nome):
    if not nome:
        return escolher_cenario("rodar a deteccao")

    for cenario in listar_cenarios():
        if cenario.nome == nome:
            print(f"Cenario selecionado: {cenario.nome}\n")
            return cenario

    raise FileNotFoundError(f"Cenario '{nome}' nao encontrado em cenarios/.")


def calcular_escala_visualizacao(frame):
    altura, largura = frame.shape[:2]
    return min(LARGURA_JANELA / largura, ALTURA_JANELA / altura)


def preparar_para_exibir(frame, escala):
    largura = int(frame.shape[1] * escala)
    altura = int(frame.shape[0] * escala)
    frame_redimensionado = cv2.resize(frame, (largura, altura), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ALTURA_JANELA, LARGURA_JANELA, 3), dtype=np.uint8)
    offset_x = (LARGURA_JANELA - largura) // 2
    offset_y = (ALTURA_JANELA - altura) // 2
    canvas[offset_y:offset_y + altura, offset_x:offset_x + largura] = frame_redimensionado
    return canvas


def carregar_base(cenario, caminho_base=None):
    caminho = Path(caminho_base) if caminho_base else cenario.imagem_base

    if not caminho.exists():
        raise FileNotFoundError(
            f"Imagem base nao encontrada: {caminho}. "
            "Use uma imagem do estacionamento vazio como base.png ou rode python selector.py para cria-la."
        )

    frame_base = cv2.imread(str(caminho))
    if frame_base is None:
        raise FileNotFoundError(f"Nao foi possivel carregar a imagem base: {caminho}")

    print(f"Imagem base carregada: {caminho}")
    return frame_base


def escalar_vagas(vagas, tamanho_origem, tamanho_destino):
    altura_origem, largura_origem = tamanho_origem[:2]
    altura_destino, largura_destino = tamanho_destino[:2]

    if (altura_origem, largura_origem) == (altura_destino, largura_destino):
        return vagas

    escala_x = largura_destino / largura_origem
    escala_y = altura_destino / altura_origem
    vagas_escaladas = []

    for vaga in vagas:
        vagas_escaladas.append([
            (int(round(x * escala_x)), int(round(y * escala_y)))
            for x, y in vaga
        ])

    return vagas_escaladas


def abrir_debug(args, cenario):
    if not args.salvar_debug:
        return None

    pasta = Path(args.debug_dir) / cenario.nome
    pasta_frames = pasta / "frames"
    pasta_recortes = pasta / "recortes"
    pasta_mascaras = pasta / "mascaras"
    pasta_diffs = pasta / "diffs"
    pasta_frames.mkdir(parents=True, exist_ok=True)
    pasta_recortes.mkdir(parents=True, exist_ok=True)
    pasta_mascaras.mkdir(parents=True, exist_ok=True)
    pasta_diffs.mkdir(parents=True, exist_ok=True)

    csv_path = pasta / "vagas_debug.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame",
            "vaga",
            "estado_final",
            "ocupada_agora",
            "decisao",
            "proporcao_mudanca",
            "media_diferenca",
            "pixels_alterados",
        ],
    )
    writer.writeheader()

    print(f"Debug sera salvo em: {pasta}")
    return {
        "frames": pasta_frames,
        "recortes": pasta_recortes,
        "mascaras": pasta_mascaras,
        "diffs": pasta_diffs,
        "csv_file": csv_file,
        "writer": writer,
    }


def fechar_debug(debug):
    if debug:
        debug["csv_file"].close()


def decidir_ocupacao_por_subtracao(evidencia, historico, config):
    proporcao = float(evidencia["proporcao_mudanca"])
    media = float(evidencia["media_diferenca"])

    if (
        proporcao >= float(config["limiar_ocupacao"])
        and media >= float(config["limiar_diferenca_media"])
    ):
        return True, "ocupada"

    if (
        proporcao <= float(config["limiar_livre"])
        or media <= float(config["limiar_diferenca_media_livre"])
    ):
        return False, "livre"

    estado_anterior = bool(historico[-1]) if historico else False
    return estado_anterior, "mantida"


def salvar_debug_vaga(debug, numero_frame, idx, evidencia, ocupada_agora, ocupada, decisao):
    if not debug:
        return

    debug["writer"].writerow(
        {
            "frame": numero_frame,
            "vaga": idx + 1,
            "estado_final": "ocupada" if ocupada else "livre",
            "ocupada_agora": int(ocupada_agora),
            "decisao": decisao,
            "proporcao_mudanca": f"{evidencia['proporcao_mudanca']:.4f}",
            "media_diferenca": f"{evidencia['media_diferenca']:.4f}",
            "pixels_alterados": evidencia["pixels_alterados"],
        }
    )


def salvar_debug_imagens(debug, numero_frame, idx, evidencia, ocupada, debug_a_cada):
    if not debug or numero_frame % max(1, debug_a_cada) != 0:
        return

    estado = "ocupada" if ocupada else "livre"
    nome = f"frame_{numero_frame:06d}_vaga_{idx + 1:03d}_{estado}"

    if evidencia["recorte"] is not None:
        cv2.imwrite(str(debug["recortes"] / f"{nome}.jpg"), evidencia["recorte"])
    if evidencia["mascara"] is not None:
        cv2.imwrite(str(debug["mascaras"] / f"{nome}.png"), evidencia["mascara"])
    if evidencia["diff"] is not None:
        cv2.imwrite(str(debug["diffs"] / f"{nome}.png"), evidencia["diff"])


def main():
    args = parse_args()
    cenario = obter_cenario(args.cenario)
    config = carregar_config_cenario(cenario, args.config)
    config_exemplo = salvar_config_exemplo(cenario)
    print(f"Exemplo de configuracao disponivel em: {config_exemplo}")

    if not cenario.video.exists():
        raise FileNotFoundError(f"Video do cenario nao encontrado: {cenario.video}")

    if not cenario.coordenadas.exists():
        raise FileNotFoundError(
            f"Coordenadas nao encontradas: {cenario.coordenadas}. Rode primeiro: python selector.py"
        )

    frame_base = carregar_base(cenario, args.base)
    with cenario.coordenadas.open("rb") as f:
        vagas_base = pickle.load(f)

    recortes_base = preparar_bases_vagas(frame_base, vagas_base, config)
    if any(recorte is None for recorte in recortes_base):
        raise RuntimeError("Nao foi possivel gerar todos os recortes da imagem base. Confira as coordenadas das vagas.")

    video = cv2.VideoCapture(str(cenario.video))
    historico_vagas = [
        deque(maxlen=int(config["janela_suavizacao"]))
        for _ in vagas_base
    ]
    debug = abrir_debug(args, cenario)

    escala_visualizacao = None
    fps_video = video.get(cv2.CAP_PROP_FPS) or 30
    tempo_ultimo_frame = cv2.getTickCount()
    numero_frame = 0
    evidencias_vagas = []
    vagas_frame = None

    print("Metodo de deteccao: subtracao entre a imagem base e o frame atual por vaga.")
    print(
        "Limiar: "
        f"{config['limiar_ocupacao']:.2f} para ocupar, "
        f"{config['limiar_livre']:.2f} para liberar, "
        f"diferenca media minima {config['limiar_diferenca_media']:.3f}."
    )

    try:
        while True:
            check, frame = video.read()
            if not check:
                break

            numero_frame += 1
            if args.limite_frames and numero_frame > args.limite_frames:
                break

            if vagas_frame is None:
                vagas_frame = escalar_vagas(vagas_base, frame_base.shape, frame.shape)
                if frame.shape[:2] != frame_base.shape[:2]:
                    print(
                        "Aviso: video e base tem resolucoes diferentes. "
                        f"Video: {frame.shape[1]}x{frame.shape[0]}, "
                        f"base: {frame_base.shape[1]}x{frame_base.shape[0]}. "
                        "As coordenadas foram escaladas para o video."
                    )

            if escala_visualizacao is None:
                escala_visualizacao = calcular_escala_visualizacao(frame)
                if not args.sem_janela:
                    cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(TITULO_JANELA, LARGURA_JANELA, ALTURA_JANELA)
                print(f"Janela da deteccao em {LARGURA_JANELA}x{ALTURA_JANELA}.")

            intervalo_analise = max(1, int(config["analisar_a_cada_n_frames"]))
            if numero_frame == 1 or numero_frame % intervalo_analise == 0:
                evidencias_vagas = analisar_vagas_por_subtracao(frame, recortes_base, vagas_frame, config)

            vagas_livres = 0
            vagas_ocupadas = 0

            for idx, vaga in enumerate(vagas_frame):
                vaga_poly = np.array(vaga, np.int32)
                evidencia = evidencias_vagas[idx]
                ocupada_agora, decisao = decidir_ocupacao_por_subtracao(
                    evidencia,
                    historico_vagas[idx],
                    config,
                )

                if not historico_vagas[idx]:
                    historico_vagas[idx].extend([ocupada_agora] * int(config["janela_suavizacao"]))
                else:
                    historico_vagas[idx].append(ocupada_agora)

                ocupada = sum(historico_vagas[idx]) >= int(config["min_ocupacoes_na_janela"])
                cor = (0, 0, 255) if ocupada else (0, 255, 0)

                if ocupada:
                    vagas_ocupadas += 1
                else:
                    vagas_livres += 1

                cv2.polylines(frame, [vaga_poly], True, cor, int(config["espessura_poligono"]))
                if bool(config["mostrar_labels_vagas"]):
                    x, y, _, _ = cv2.boundingRect(vaga_poly)
                    label = (
                        f"V{idx + 1} {decisao[:1].upper()} diff {evidencia['proporcao_mudanca']:.2f} "
                        f"med {evidencia['media_diferenca']:.2f}"
                    )
                    cv2.putText(frame, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1)

                salvar_debug_vaga(debug, numero_frame, idx, evidencia, ocupada_agora, ocupada, decisao)
                salvar_debug_imagens(debug, numero_frame, idx, evidencia, ocupada, args.debug_a_cada)

            cv2.putText(frame, f"Vagas Livres: {vagas_livres}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(frame, f"Vagas Ocupadas: {vagas_ocupadas}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            if debug and numero_frame % max(1, args.debug_a_cada) == 0:
                cv2.imwrite(str(debug["frames"] / f"frame_{numero_frame:06d}.jpg"), frame)

            if not args.sem_janela:
                cv2.imshow(TITULO_JANELA, preparar_para_exibir(frame, escala_visualizacao))

            tempo_frame_ms = 1000 / fps_video
            tempo_agora = cv2.getTickCount()
            tempo_processado_ms = ((tempo_agora - tempo_ultimo_frame) / cv2.getTickFrequency()) * 1000
            tempo_ultimo_frame = tempo_agora
            espera_ms = max(1, int(tempo_frame_ms - tempo_processado_ms))

            if config["pular_frames"] and tempo_processado_ms > tempo_frame_ms:
                frames_para_pular = int(tempo_processado_ms // tempo_frame_ms) - 1
                for _ in range(max(0, frames_para_pular)):
                    video.grab()

            if not args.sem_janela and cv2.waitKey(espera_ms) & 0xFF == ord("q"):
                break
    finally:
        video.release()
        fechar_debug(debug)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
