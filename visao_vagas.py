import json
from pathlib import Path

import cv2
import numpy as np


CONFIG_PADRAO = {
    "limiar_pixels_diferentes": 28,
    "limiar_ocupacao": 0.18,
    "limiar_livre": 0.10,
    "limiar_diferenca_media": 0.035,
    "limiar_diferenca_media_livre": 0.020,
    "normalizar_iluminacao": True,
    "blur_kernel": 5,
    "morfologia_kernel": 3,
    "morfologia_iteracoes": 1,
    "janela_suavizacao": 5,
    "min_ocupacoes_na_janela": 4,
    "analisar_a_cada_n_frames": 2,
    "pular_frames": True,
    "tamanho_recorte_vaga": 160,
    "mostrar_labels_vagas": True,
    "espessura_poligono": 2,
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


def _kernel_impar(valor, minimo=1):
    valor = max(minimo, int(valor))
    return valor if valor % 2 == 1 else valor + 1


def preparar_recorte_subtracao(recorte, config):
    if recorte is None or recorte.size == 0:
        return None

    cinza = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    blur_kernel = _kernel_impar(config.get("blur_kernel", 5))
    if blur_kernel > 1:
        cinza = cv2.GaussianBlur(cinza, (blur_kernel, blur_kernel), 0)

    return cinza


def analisar_subtracao_recorte(recorte_atual, recorte_base, config):
    if recorte_atual is None or recorte_base is None:
        return {
            "ocupada": False,
            "proporcao_mudanca": 0.0,
            "media_diferenca": 0.0,
            "pixels_alterados": 0,
            "mascara": None,
            "diff": None,
        }

    atual = preparar_recorte_subtracao(recorte_atual, config)
    base = preparar_recorte_subtracao(recorte_base, config)
    if atual is None or base is None:
        return {
            "ocupada": False,
            "proporcao_mudanca": 0.0,
            "media_diferenca": 0.0,
            "pixels_alterados": 0,
            "mascara": None,
            "diff": None,
        }

    if bool(config.get("normalizar_iluminacao", True)):
        ajuste = float(np.mean(base)) - float(np.mean(atual))
        atual = np.clip(atual.astype(np.float32) + ajuste, 0, 255).astype(np.uint8)

    diff = cv2.absdiff(base, atual)
    _, mascara = cv2.threshold(
        diff,
        int(config["limiar_pixels_diferentes"]),
        255,
        cv2.THRESH_BINARY,
    )

    kernel_tamanho = int(config.get("morfologia_kernel", 3))
    if kernel_tamanho > 1:
        kernel_tamanho = _kernel_impar(kernel_tamanho)
        kernel = np.ones((kernel_tamanho, kernel_tamanho), np.uint8)
        iteracoes = max(1, int(config.get("morfologia_iteracoes", 1)))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=iteracoes)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=iteracoes)

    pixels_alterados = cv2.countNonZero(mascara)
    area = float(mascara.shape[0] * mascara.shape[1])
    proporcao_mudanca = pixels_alterados / area if area else 0.0
    media_diferenca = float(np.mean(diff)) / 255.0
    ocupada = (
        proporcao_mudanca >= float(config["limiar_ocupacao"])
        and media_diferenca >= float(config["limiar_diferenca_media"])
    )

    return {
        "ocupada": ocupada,
        "proporcao_mudanca": proporcao_mudanca,
        "media_diferenca": media_diferenca,
        "pixels_alterados": pixels_alterados,
        "mascara": mascara,
        "diff": diff,
    }


def preparar_bases_vagas(frame_base, vagas, config):
    tamanho_recorte = int(config["tamanho_recorte_vaga"])
    return [
        recortar_vaga_normalizada(frame_base, vaga, tamanho=tamanho_recorte)
        for vaga in vagas
    ]


def analisar_vagas_por_subtracao(frame, recortes_base, vagas, config):
    evidencias = []
    tamanho_recorte = int(config["tamanho_recorte_vaga"])

    for vaga, recorte_base in zip(vagas, recortes_base):
        recorte_atual = recortar_vaga_normalizada(frame, vaga, tamanho=tamanho_recorte)
        subtracao = analisar_subtracao_recorte(recorte_atual, recorte_base, config)
        subtracao["recorte"] = recorte_atual
        evidencias.append(subtracao)

    return evidencias
