import argparse
import csv
import pickle
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from cenarios import escolher_cenario, listar_cenarios
from visao_vagas import (
    analisar_linhas_vaga,
    associar_veiculos_vagas,
    carregar_config_cenario,
    recortar_vaga_normalizada,
    salvar_config_exemplo,
)


TITULO_JANELA = "Deteccao com YOLO"
LARGURA_JANELA = 1280
ALTURA_JANELA = 720


def parse_args():
    parser = argparse.ArgumentParser(description="Detecta vagas livres e ocupadas em um cenario.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--config", help="Caminho para um config.json alternativo.")
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


def preparar_para_inferencia(frame, largura_inferencia):
    altura, largura = frame.shape[:2]
    if largura <= largura_inferencia:
        return frame, 1.0, 1.0

    escala = largura_inferencia / largura
    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    frame_redimensionado = cv2.resize(frame, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)
    escala_x = largura / nova_largura
    escala_y = altura / nova_altura
    return frame_redimensionado, escala_x, escala_y


def converter_caixa_para_original(caixa, escala_x, escala_y):
    x1, y1, x2, y2 = caixa
    return (
        int(x1 * escala_x),
        int(y1 * escala_y),
        int(x2 * escala_x),
        int(y2 * escala_y),
    )


def analisar_recortes_vagas(frame, vagas, classificador_vagas, config):
    evidencias = []
    recortes = []
    indices_validos = []
    tamanho_recorte = int(config["tamanho_recorte_vaga"])

    for idx, vaga in enumerate(vagas):
        recorte = recortar_vaga_normalizada(frame, vaga, tamanho=tamanho_recorte)
        linhas = analisar_linhas_vaga(recorte)
        evidencias.append(
            {
                "recorte": recorte,
                "ocupada_cls": False,
                "confianca_cls": 0.0,
                "classe_cls": "sem_modelo" if classificador_vagas is None else "sem_recorte",
                "linhas": linhas,
            }
        )

        if recorte is not None:
            recortes.append(recorte)
            indices_validos.append(idx)

    if classificador_vagas is None or not recortes:
        return evidencias

    resultados = classificador_vagas.predict(recortes, imgsz=tamanho_recorte, verbose=False)
    for idx, resultado in zip(indices_validos, resultados):
        nomes = resultado.names
        classe_id = int(resultado.probs.top1)
        confianca = float(resultado.probs.top1conf)
        classe_nome = nomes[classe_id].lower()
        evidencias[idx].update(
            {
                "ocupada_cls": classe_nome == "occupied" and confianca >= float(config["confianca_ocupada"]),
                "confianca_cls": confianca,
                "classe_cls": classe_nome,
            }
        )

    return evidencias


def decidir_ocupacao(evidencia, associacao, historico, config):
    sobreposicao_yolo = associacao["sobreposicao"]
    yolo_na_vaga = associacao["ponto_inferior_na_vaga"]
    ocupada_yolo = (
        (yolo_na_vaga and sobreposicao_yolo >= float(config["limiar_sobreposicao"]))
        or sobreposicao_yolo >= float(config["limiar_sobreposicao_sem_ponto_base"])
    )

    classe_cls = evidencia["classe_cls"]
    confianca_cls = evidencia["confianca_cls"]
    ocupada_cls = classe_cls == "occupied" and confianca_cls >= float(config["confianca_ocupada"])
    permite_ocupada_sem_yolo = bool(config["permitir_ocupada_sem_yolo"])
    ocupada_cls_sem_yolo = permite_ocupada_sem_yolo and (
        classe_cls == "occupied" and confianca_cls >= float(config["confianca_ocupada_sem_yolo"])
    )
    livre_cls_forte = classe_cls == "empty" and confianca_cls >= float(config["confianca_livre"])
    escore_linhas = evidencia["linhas"]["escore"]
    linhas_livre = escore_linhas >= float(config["limiar_linhas_livre"])
    linhas_livre_forte = escore_linhas >= float(config["limiar_linhas_livre_forte"])

    if livre_cls_forte and not yolo_na_vaga:
        return False, "C", ocupada_yolo, linhas_livre

    if linhas_livre_forte and not yolo_na_vaga:
        return False, "L", ocupada_yolo, linhas_livre

    if ocupada_cls and (ocupada_yolo or ocupada_cls_sem_yolo):
        return True, "C", ocupada_yolo, linhas_livre

    if ocupada_cls and linhas_livre:
        margem = float(config["margem_desempate_classificador"])
        if escore_linhas + margem >= confianca_cls and not yolo_na_vaga:
            return False, "L", ocupada_yolo, linhas_livre

    if ocupada_yolo and yolo_na_vaga:
        return True, "Y", ocupada_yolo, linhas_livre

    if linhas_livre:
        return False, "L", ocupada_yolo, linhas_livre

    estado_anterior = bool(historico[-1]) if historico else False
    return estado_anterior, "-", ocupada_yolo, linhas_livre


def abrir_debug(args, cenario):
    if not args.salvar_debug:
        return None

    pasta = Path(args.debug_dir) / cenario.nome
    pasta_frames = pasta / "frames"
    pasta_recortes = pasta / "recortes"
    pasta_frames.mkdir(parents=True, exist_ok=True)
    pasta_recortes.mkdir(parents=True, exist_ok=True)

    csv_path = pasta / "vagas_debug.csv"
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame",
            "vaga",
            "estado_final",
            "ocupada_agora",
            "origem",
            "classe_classificador",
            "confianca_classificador",
            "ocupada_classificador",
            "sobreposicao_yolo",
            "confianca_yolo",
            "ponto_inferior_na_vaga",
            "linhas_livre",
            "escore_linhas",
            "densidade_marcacoes",
            "densidade_linhas",
            "quantidade_linhas",
        ],
    )
    writer.writeheader()

    print(f"Debug sera salvo em: {pasta}")
    return {
        "pasta": pasta,
        "frames": pasta_frames,
        "recortes": pasta_recortes,
        "csv_file": csv_file,
        "writer": writer,
    }


def fechar_debug(debug):
    if debug:
        debug["csv_file"].close()


def salvar_debug_vaga(debug, numero_frame, idx, evidencia, associacao, ocupada_agora, ocupada, origem, linhas_livre):
    if not debug:
        return

    linhas = evidencia["linhas"]
    debug["writer"].writerow(
        {
            "frame": numero_frame,
            "vaga": idx + 1,
            "estado_final": "ocupada" if ocupada else "livre",
            "ocupada_agora": int(ocupada_agora),
            "origem": origem,
            "classe_classificador": evidencia["classe_cls"],
            "confianca_classificador": f"{evidencia['confianca_cls']:.4f}",
            "ocupada_classificador": int(evidencia["ocupada_cls"]),
            "sobreposicao_yolo": f"{associacao['sobreposicao']:.4f}",
            "confianca_yolo": f"{associacao['confianca']:.4f}",
            "ponto_inferior_na_vaga": int(associacao["ponto_inferior_na_vaga"]),
            "linhas_livre": int(linhas_livre),
            "escore_linhas": f"{linhas['escore']:.4f}",
            "densidade_marcacoes": f"{linhas['densidade_marcacoes']:.4f}",
            "densidade_linhas": f"{linhas['densidade_linhas']:.4f}",
            "quantidade_linhas": linhas["quantidade_linhas"],
        }
    )


def main():
    args = parse_args()
    cenario = obter_cenario(args.cenario)
    config = carregar_config_cenario(cenario, args.config)
    config_exemplo = salvar_config_exemplo(cenario)
    print(f"Exemplo de configuracao disponivel em: {config_exemplo}")

    arquivo_coordenadas = cenario.coordenadas
    video_entrada = cenario.video

    if not video_entrada.exists():
        raise FileNotFoundError(f"Video do cenario nao encontrado: {video_entrada}")

    if not arquivo_coordenadas.exists():
        raise FileNotFoundError(
            f"Coordenadas nao encontradas: {arquivo_coordenadas}. Rode primeiro: python selector.py"
        )

    model = YOLO(config["modelo_yolo"])
    caminho_classificador = Path(config["modelo_classificador_vagas"])
    classificador_vagas = YOLO(str(caminho_classificador)) if caminho_classificador.exists() else None
    if classificador_vagas is None:
        print(f"Classificador de vagas nao encontrado em {caminho_classificador}. Usando YOLO + linhas.")
    else:
        print(f"Classificador de vagas carregado: {caminho_classificador}")

    with arquivo_coordenadas.open("rb") as f:
        vagas_pos = pickle.load(f)

    video = cv2.VideoCapture(str(video_entrada))
    historico_vagas = [
        deque(maxlen=int(config["janela_suavizacao"]))
        for _ in vagas_pos
    ]
    debug = abrir_debug(args, cenario)

    escala_visualizacao = None
    fps_video = video.get(cv2.CAP_PROP_FPS) or 30
    tempo_ultimo_frame = cv2.getTickCount()
    numero_frame = 0
    deteccoes_atuais = []
    evidencias_vagas = analisar_recortes_vagas(
        np.zeros((10, 10, 3), dtype=np.uint8),
        [],
        classificador_vagas,
        config,
    )
    associacoes_vagas = []

    try:
        while True:
            check, frame = video.read()
            if not check:
                break

            numero_frame += 1
            if args.limite_frames and numero_frame > args.limite_frames:
                break

            if escala_visualizacao is None:
                escala_visualizacao = calcular_escala_visualizacao(frame)
                if not args.sem_janela:
                    cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(TITULO_JANELA, LARGURA_JANELA, ALTURA_JANELA)
                print(f"Janela da deteccao em {LARGURA_JANELA}x{ALTURA_JANELA}. Processamento no tamanho original.")

            intervalo_deteccao = max(1, int(config["detectar_a_cada_n_frames"]))
            if numero_frame == 1 or numero_frame % intervalo_deteccao == 0:
                frame_inferencia, escala_x_inferencia, escala_y_inferencia = preparar_para_inferencia(
                    frame,
                    int(config["largura_inferencia"]),
                )

                results = model.predict(
                    frame_inferencia,
                    classes=config["classes_veiculos"],
                    conf=float(config["confianca_minima_yolo"]),
                    iou=float(config["iou_yolo"]),
                    verbose=False,
                )

                deteccoes_atuais = []
                evidencias_vagas = analisar_recortes_vagas(frame, vagas_pos, classificador_vagas, config)

                for r in results:
                    for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                        caixa_original = converter_caixa_para_original(
                            tuple(map(int, box)),
                            escala_x_inferencia,
                            escala_y_inferencia,
                        )
                        deteccoes_atuais.append((caixa_original, float(conf)))

                associacoes_vagas = associar_veiculos_vagas(
                    vagas_pos,
                    deteccoes_atuais,
                    frame.shape,
                    limiar_associacao=float(config["limiar_sobreposicao_associacao"]),
                )

            for caixa_original, conf in deteccoes_atuais:
                x1, y1, x2, y2 = caixa_original
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"VEICULO {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            vagas_livres = 0
            vagas_ocupadas = 0

            for idx, vaga in enumerate(vagas_pos):
                vaga_poly = np.array(vaga, np.int32)
                evidencia = evidencias_vagas[idx]
                associacao = associacoes_vagas[idx] if associacoes_vagas else {
                    "sobreposicao": 0.0,
                    "score": 0.0,
                    "confianca": 0.0,
                    "ponto_inferior_na_vaga": False,
                    "caixa": None,
                }

                ocupada_agora, origem, ocupada_yolo, linhas_livre = decidir_ocupacao(
                    evidencia,
                    associacao,
                    historico_vagas[idx],
                    config,
                )
                if not historico_vagas[idx]:
                    historico_vagas[idx].extend([ocupada_agora] * int(config["janela_suavizacao"]))
                else:
                    historico_vagas[idx].append(ocupada_agora)
                ocupada = sum(historico_vagas[idx]) >= int(config["min_ocupacoes_na_janela"])

                color = (0, 0, 255) if ocupada else (0, 255, 0)
                if ocupada:
                    vagas_ocupadas += 1
                else:
                    vagas_livres += 1

                cv2.polylines(frame, [vaga_poly], True, color, 2)
                x, y, _, _ = cv2.boundingRect(vaga_poly)
                label = (
                    f"V{idx + 1} {origem} {evidencia['classe_cls'][:3]} {evidencia['confianca_cls']:.2f} "
                    f"ov {associacao['sobreposicao']:.2f} ln {evidencia['linhas']['escore']:.2f}"
                )
                cv2.putText(frame, label, (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

                salvar_debug_vaga(
                    debug,
                    numero_frame,
                    idx,
                    evidencia,
                    associacao,
                    ocupada_agora,
                    ocupada,
                    origem,
                    linhas_livre,
                )

                if (
                    debug
                    and numero_frame % max(1, args.debug_a_cada) == 0
                    and evidencia["recorte"] is not None
                ):
                    nome_recorte = (
                        f"frame_{numero_frame:06d}_vaga_{idx + 1:03d}_"
                        f"{'ocupada' if ocupada else 'livre'}_{origem}.jpg"
                    )
                    cv2.imwrite(str(debug["recortes"] / nome_recorte), evidencia["recorte"])

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
