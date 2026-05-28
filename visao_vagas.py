import json
from pathlib import Path

import cv2
import numpy as np


CONFIG_PADRAO = {
    "modelo_yolo": "yolo11s.pt",
    "modelo_classificador_vagas": "models/parking_occupancy_yolo11n_cls.pt",
    "classes_veiculos": [2, 3, 5, 7],
    "confianca_minima_yolo": 0.30,
    "iou_yolo": 0.50,
    "limiar_sobreposicao": 0.45,
    "limiar_sobreposicao_sem_ponto_base": 0.75,
    "limiar_sobreposicao_associacao": 0.08,
    "limiar_linhas_livre": 0.18,
    "limiar_linhas_livre_forte": 0.65,
    "confianca_ocupada": 0.96,
    "confianca_ocupada_sem_yolo": 0.995,
    "confianca_livre": 0.90,
    "confianca_livre_fraca": 0.50,
    "margem_desempate_classificador": 0.08,
    "permitir_ocupada_sem_yolo": False,
    "liberar_sem_evidencia_de_veiculo": True,
    "janela_suavizacao": 5,
    "min_ocupacoes_na_janela": 4,
    "largura_inferencia": 960,
    "detectar_a_cada_n_frames": 3,
    "pular_frames": True,
    "tamanho_recorte_vaga": 160,
}


def carregar_config_cenario(cenario, caminho_config=None):
    config = CONFIG_PADRAO.copy()
    caminho = Path(caminho_config) if caminho_config else cenario.pasta / "config.json"

    if caminho.exists():
        with caminho.open("r", encoding="utf-8") as f:
            config_usuario = json.load(f)
        config.update(config_usuario)
        print(f"Configuracao carregada: {caminho}")
    else:
        print(f"Configuracao especifica nao encontrada. Usando padrao interno.")

    return config


def salvar_config_exemplo(cenario):
    caminho = cenario.pasta / "config.example.json"
    if caminho.exists():
        return caminho

    with caminho.open("w", encoding="utf-8") as f:
        json.dump(CONFIG_PADRAO, f, indent=2)
        f.write("\n")

    return caminho


def ordenar_pontos_quadrilatero(pontos):
    pontos = np.array(pontos, dtype=np.float32)

    soma = pontos.sum(axis=1)
    diferenca = np.diff(pontos, axis=1).reshape(-1)

    ordenados = np.zeros((4, 2), dtype=np.float32)
    ordenados[0] = pontos[np.argmin(soma)]
    ordenados[2] = pontos[np.argmax(soma)]
    ordenados[1] = pontos[np.argmin(diferenca)]
    ordenados[3] = pontos[np.argmax(diferenca)]

    return ordenados


def recortar_vaga_normalizada(frame, vaga, tamanho=160):
    if len(vaga) != 4:
        return None

    origem = ordenar_pontos_quadrilatero(vaga)
    destino = np.array(
        [
            [0, 0],
            [tamanho - 1, 0],
            [tamanho - 1, tamanho - 1],
            [0, tamanho - 1],
        ],
        dtype=np.float32,
    )

    matriz = cv2.getPerspectiveTransform(origem, destino)
    recorte = cv2.warpPerspective(frame, matriz, (tamanho, tamanho))

    if recorte.size == 0:
        return None

    return recorte


def recortar_vaga_mascarada(frame, vaga_poly):
    altura_frame, largura_frame = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(vaga_poly)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(largura_frame, x + w)
    y2 = min(altura_frame, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    recorte = frame[y1:y2, x1:x2]
    if recorte.size == 0:
        return None

    vaga_roi = vaga_poly - np.array([x1, y1])
    mascara = np.zeros((recorte.shape[0], recorte.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mascara, [vaga_roi], 255)
    return cv2.bitwise_and(recorte, recorte, mask=mascara)


def analisar_linhas_vaga(recorte):
    if recorte is None or recorte.size == 0:
        return {
            "escore": 0.0,
            "densidade_marcacoes": 0.0,
            "densidade_linhas": 0.0,
            "quantidade_linhas": 0,
        }

    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    mascara_branco = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([180, 95, 255]))
    mascara_amarelo = cv2.inRange(hsv, np.array([15, 40, 100]), np.array([45, 255, 255]))
    mascara = cv2.bitwise_or(mascara_branco, mascara_amarelo)

    kernel = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    bordas = cv2.Canny(mascara, 50, 150)
    altura, largura = recorte.shape[:2]
    comprimento_minimo = max(12, int(min(altura, largura) * 0.18))
    linhas = cv2.HoughLinesP(
        bordas,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=comprimento_minimo,
        maxLineGap=8,
    )

    mascara_linhas = np.zeros((altura, largura), dtype=np.uint8)
    quantidade_linhas = 0
    if linhas is not None:
        quantidade_linhas = len(linhas)
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(mascara_linhas, (x1, y1), (x2, y2), 255, 2)

    area = float(altura * largura)
    densidade_marcacoes = cv2.countNonZero(mascara) / area
    densidade_linhas = cv2.countNonZero(mascara_linhas) / area

    escore = min(1.0, (densidade_marcacoes * 2.5) + (densidade_linhas * 10.0))
    return {
        "escore": escore,
        "densidade_marcacoes": densidade_marcacoes,
        "densidade_linhas": densidade_linhas,
        "quantidade_linhas": quantidade_linhas,
    }


def centro_da_caixa(caixa):
    x1, y1, x2, y2 = caixa
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def ponto_inferior_central_caixa(caixa):
    x1, _, x2, y2 = caixa
    return int((x1 + x2) / 2), int(y2)


def proporcao_sobreposta(vaga_poly, caixa, tamanho_frame):
    altura_frame, largura_frame = tamanho_frame[:2]
    x1, y1, x2, y2 = caixa
    x1 = max(0, min(largura_frame - 1, x1))
    x2 = max(0, min(largura_frame - 1, x2))
    y1 = max(0, min(altura_frame - 1, y1))
    y2 = max(0, min(altura_frame - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    vx, vy, vw, vh = cv2.boundingRect(vaga_poly)
    rx1 = max(0, min(x1, vx))
    ry1 = max(0, min(y1, vy))
    rx2 = min(largura_frame - 1, max(x2, vx + vw))
    ry2 = min(altura_frame - 1, max(y2, vy + vh))

    largura_roi = rx2 - rx1
    altura_roi = ry2 - ry1
    if largura_roi <= 0 or altura_roi <= 0:
        return 0.0

    vaga_roi = vaga_poly - np.array([rx1, ry1])
    caixa_roi = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)

    mascara_vaga = np.zeros((altura_roi, largura_roi), dtype=np.uint8)
    mascara_caixa = np.zeros((altura_roi, largura_roi), dtype=np.uint8)

    cv2.fillPoly(mascara_vaga, [vaga_roi], 255)
    cv2.rectangle(mascara_caixa, caixa_roi[:2], caixa_roi[2:], 255, -1)

    area_vaga = cv2.countNonZero(mascara_vaga)
    if area_vaga == 0:
        return 0.0

    area_intersecao = cv2.countNonZero(cv2.bitwise_and(mascara_vaga, mascara_caixa))
    return area_intersecao / area_vaga


def associar_veiculos_vagas(vagas, deteccoes, tamanho_frame, limiar_associacao=0.08):
    associacoes = [
        {
            "sobreposicao": 0.0,
            "score": 0.0,
            "confianca": 0.0,
            "ponto_inferior_na_vaga": False,
            "caixa": None,
        }
        for _ in vagas
    ]

    vagas_poly = [np.array(vaga, np.int32) for vaga in vagas]

    for caixa, confianca in deteccoes:
        melhor_idx = None
        melhor_score = 0.0
        melhor_sobreposicao = 0.0
        melhor_ponto_na_vaga = False
        ponto_base = ponto_inferior_central_caixa(caixa)

        for idx, vaga_poly in enumerate(vagas_poly):
            sobreposicao = proporcao_sobreposta(vaga_poly, caixa, tamanho_frame)
            ponto_na_vaga = cv2.pointPolygonTest(vaga_poly, ponto_base, False) >= 0
            score = sobreposicao + (0.20 if ponto_na_vaga else 0.0)

            if score > melhor_score:
                melhor_idx = idx
                melhor_score = score
                melhor_sobreposicao = sobreposicao
                melhor_ponto_na_vaga = ponto_na_vaga

        if melhor_idx is None:
            continue

        if melhor_sobreposicao < limiar_associacao and not melhor_ponto_na_vaga:
            continue

        if melhor_score > associacoes[melhor_idx]["score"]:
            associacoes[melhor_idx] = {
                "sobreposicao": melhor_sobreposicao,
                "score": melhor_score,
                "confianca": confianca,
                "ponto_inferior_na_vaga": melhor_ponto_na_vaga,
                "caixa": caixa,
            }

    return associacoes
