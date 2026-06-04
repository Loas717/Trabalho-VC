import argparse
import cv2
import pickle
import numpy as np
from pathlib import Path

from cenarios import escolher_cenario, listar_cenarios

TITULO_JANELA = "Editor de vagas"
RAIO_SELECAO = 12
LARGURA_JANELA = 1280
ALTURA_JANELA = 720
PASSO_MOVIMENTO = 2
PASSO_MOVIMENTO_RAPIDO = 10

def parse_args():
    parser = argparse.ArgumentParser(description="Editor de poligonos das vagas.")
    parser.add_argument("--cenario", help="Nome da pasta dentro de cenarios/ para evitar o menu interativo.")
    parser.add_argument("--base", help="Imagem base alternativa para marcar as vagas.")
    return parser.parse_args()


def obter_cenario(nome):
    if not nome:
        return escolher_cenario("editar as vagas")

    for cenario in listar_cenarios():
        if cenario.nome == nome:
            print(f"Cenario selecionado: {cenario.nome}\n")
            return cenario

    raise FileNotFoundError(f"Cenario '{nome}' nao encontrado em cenarios/.")


args = parse_args()
cenario = obter_cenario(args.cenario)
ARQUIVO_COORDENADAS = cenario.coordenadas
IMAGEM_BASE = Path(args.base) if args.base else cenario.imagem_base
BASE_CUSTOMIZADA = args.base is not None
VIDEO_ENTRADA = cenario.video

vagas = []
vaga_atual = []
ponto_arrastando = None
vaga_selecionada = None
mouse_pos = (0, 0)
escala_visualizacao = 1.0
offset_x_visualizacao = 0
offset_y_visualizacao = 0


def carregar_vagas():
    if not ARQUIVO_COORDENADAS.exists():
        print("Arquivo de coordenadas nao encontrado. Criando nova marcacao.")
        return []

    with ARQUIVO_COORDENADAS.open('rb') as f:
        dados = pickle.load(f)

    print(f"{len(dados)} vaga(s) carregada(s) de {ARQUIVO_COORDENADAS}.")
    return dados


def salvar_vagas():
    with ARQUIVO_COORDENADAS.open('wb') as f:
        pickle.dump(vagas, f)
    print(f"{len(vagas)} vaga(s) salva(s) em {ARQUIVO_COORDENADAS}.")


def criar_imagem_base_pelo_video():
    if IMAGEM_BASE.exists():
        return

    if BASE_CUSTOMIZADA:
        raise FileNotFoundError(f"Imagem base alternativa nao encontrada: {IMAGEM_BASE}")

    if not VIDEO_ENTRADA.exists():
        raise FileNotFoundError(
            f"Imagem base nao encontrada ({IMAGEM_BASE}) e video nao encontrado ({VIDEO_ENTRADA})."
        )

    video = cv2.VideoCapture(str(VIDEO_ENTRADA))
    check, frame = video.read()
    video.release()

    if not check:
        raise RuntimeError(f"Nao foi possivel ler o primeiro frame do video: {VIDEO_ENTRADA}")

    IMAGEM_BASE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(IMAGEM_BASE), frame)
    print(f"Imagem base criada automaticamente em {IMAGEM_BASE}.")


def encontrar_ponto_proximo(x, y):
    for vaga_idx, vaga in enumerate(vagas):
        for ponto_idx, (px, py) in enumerate(vaga):
            distancia = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if distancia <= RAIO_SELECAO:
                return vaga_idx, ponto_idx
    return None


def encontrar_vaga_sob_mouse(x, y):
    for vaga_idx, vaga in enumerate(vagas):
        vaga_poly = np.array(vaga, np.int32)
        if cv2.pointPolygonTest(vaga_poly, (x, y), False) >= 0:
            return vaga_idx
    return None


def limitar_ponto_imagem(x, y):
    largura_original = img.shape[1] - 1
    altura_original = img.shape[0] - 1
    return (
        max(0, min(largura_original, int(round(x)))),
        max(0, min(altura_original, int(round(y)))),
    )


def selecionar_vaga(vaga_idx):
    global vaga_selecionada
    vaga_selecionada = vaga_idx
    if vaga_idx is not None:
        print(f"Vaga {vaga_idx + 1} selecionada.")


def mover_vaga(vaga_idx, dx, dy):
    if vaga_idx is None or not (0 <= vaga_idx < len(vagas)):
        print("Nenhuma vaga selecionada para mover.")
        return

    vagas[vaga_idx] = [
        limitar_ponto_imagem(px + dx, py + dy)
        for px, py in vagas[vaga_idx]
    ]


def copiar_vaga_para_mouse():
    global vaga_selecionada
    origem_idx = vaga_selecionada

    if origem_idx is None:
        origem_idx = encontrar_vaga_sob_mouse(*mouse_pos)

    if origem_idx is None and vagas:
        origem_idx = len(vagas) - 1

    if origem_idx is None:
        print("Nenhuma vaga para copiar. Marque uma vaga primeiro.")
        return

    vaga = vagas[origem_idx]
    centro = np.mean(np.array(vaga, dtype=np.float32), axis=0)
    dx = mouse_pos[0] - centro[0]
    dy = mouse_pos[1] - centro[1]
    nova_vaga = [
        limitar_ponto_imagem(px + dx, py + dy)
        for px, py in vaga
    ]

    vagas.append(nova_vaga)
    vaga_selecionada = len(vagas) - 1
    print(f"Vaga {origem_idx + 1} copiada para a posicao do mouse. Total: {len(vagas)}")


def apagar_vaga(vaga_idx):
    global vaga_selecionada
    if vaga_idx is None or not (0 <= vaga_idx < len(vagas)):
        print("Nenhuma vaga selecionada para apagar.")
        return

    vagas.pop(vaga_idx)
    if not vagas:
        vaga_selecionada = None
    elif vaga_idx >= len(vagas):
        vaga_selecionada = len(vagas) - 1
    else:
        vaga_selecionada = vaga_idx
    print(f"Vaga {vaga_idx + 1} removida.")


def deslocamento_tecla(tecla):
    tecla_ascii = tecla & 0xFF
    passo = PASSO_MOVIMENTO_RAPIDO if chr(tecla_ascii).isupper() else PASSO_MOVIMENTO

    movimentos_ascii = {
        ord("j"): (-PASSO_MOVIMENTO, 0),
        ord("J"): (-PASSO_MOVIMENTO_RAPIDO, 0),
        ord("l"): (PASSO_MOVIMENTO, 0),
        ord("L"): (PASSO_MOVIMENTO_RAPIDO, 0),
        ord("i"): (0, -PASSO_MOVIMENTO),
        ord("I"): (0, -PASSO_MOVIMENTO_RAPIDO),
        ord("k"): (0, PASSO_MOVIMENTO),
        ord("K"): (0, PASSO_MOVIMENTO_RAPIDO),
    }
    if tecla_ascii in movimentos_ascii:
        return movimentos_ascii[tecla_ascii]

    movimentos_setas = {
        2424832: (-passo, 0),
        2555904: (passo, 0),
        2490368: (0, -passo),
        2621440: (0, passo),
        81: (-passo, 0),
        83: (passo, 0),
        82: (0, -passo),
        84: (0, passo),
    }
    return movimentos_setas.get(tecla)


def calcular_escala_visualizacao(img):
    altura, largura = img.shape[:2]
    return min(LARGURA_JANELA / largura, ALTURA_JANELA / altura)


def converter_mouse_para_original(x, y):
    x_sem_borda = x - offset_x_visualizacao
    y_sem_borda = y - offset_y_visualizacao
    x_original = int(x_sem_borda / escala_visualizacao)
    y_original = int(y_sem_borda / escala_visualizacao)
    largura_original = img.shape[1] - 1
    altura_original = img.shape[0] - 1
    return (
        max(0, min(largura_original, x_original)),
        max(0, min(altura_original, y_original)),
    )


def preparar_para_exibir(img):
    largura = int(img.shape[1] * escala_visualizacao)
    altura = int(img.shape[0] * escala_visualizacao)
    img_redimensionada = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ALTURA_JANELA, LARGURA_JANELA, 3), dtype=np.uint8)
    canvas[
        offset_y_visualizacao:offset_y_visualizacao + altura,
        offset_x_visualizacao:offset_x_visualizacao + largura
    ] = img_redimensionada
    return canvas


def desenhar_texto_com_contorno(img, texto, origem, escala, cor=(255, 255, 255)):
    cv2.putText(img, texto, origem, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), max(2, int(escala * 5)))
    cv2.putText(img, texto, origem, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, max(1, int(escala * 2)))


def quebrar_texto_por_largura(texto, escala, largura_maxima):
    palavras = texto.split()
    linhas = []
    linha_atual = ""

    for palavra in palavras:
        candidata = palavra if not linha_atual else f"{linha_atual} {palavra}"
        largura_texto = cv2.getTextSize(candidata, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)[0][0]

        if largura_texto <= largura_maxima or not linha_atual:
            linha_atual = candidata
        else:
            linhas.append(linha_atual)
            linha_atual = palavra

    if linha_atual:
        linhas.append(linha_atual)

    return linhas


def desenhar_instrucoes(img):
    instrucoes = [
        "Clique vazio: novo ponto | Clique na vaga: selecionar | Arraste ponto amarelo: ajustar",
        "C: copiar selecionada para o mouse | Setas/IJKL: mover | IJKL maiusculo acelera",
        "Botao direito ou S: salvar | D: apagar selecionada/sob mouse | U: desfazer | Q: sair",
        f"Vagas: {len(vagas)} | Selecionada: {vaga_selecionada + 1 if vaga_selecionada is not None else '-'} | Nova vaga: {len(vaga_atual)}/4",
    ]

    escala_texto = max(0.45, min(0.85, LARGURA_JANELA / 1500))
    margem = max(8, int(LARGURA_JANELA * 0.01))
    y = margem + int(24 * escala_texto)
    altura_linha = int(34 * escala_texto)
    largura_texto = LARGURA_JANELA - (margem * 2)

    for instrucao in instrucoes:
        for linha in quebrar_texto_por_largura(instrucao, escala_texto, largura_texto):
            desenhar_texto_com_contorno(img, linha, (margem, y), escala_texto)
            y += altura_linha

    return img


def mouse_click(event, x, y, flags, params):
    global vaga_atual, ponto_arrastando, mouse_pos
    x_original, y_original = converter_mouse_para_original(x, y)
    mouse_pos = (x_original, y_original)

    if event == cv2.EVENT_LBUTTONDOWN:
        ponto_proximo = encontrar_ponto_proximo(x_original, y_original)
        if ponto_proximo:
            ponto_arrastando = ponto_proximo
            selecionar_vaga(ponto_proximo[0])
            return

        vaga_idx = encontrar_vaga_sob_mouse(x_original, y_original)
        if vaga_idx is not None and not vaga_atual:
            selecionar_vaga(vaga_idx)
            return

        vaga_atual.append((x_original, y_original))
        if len(vaga_atual) == 4:
            vagas.append(vaga_atual.copy())
            selecionar_vaga(len(vagas) - 1)
            vaga_atual = []
            print(f"Nova vaga adicionada. Total: {len(vagas)}")

    elif event == cv2.EVENT_MOUSEMOVE and ponto_arrastando:
        vaga_idx, ponto_idx = ponto_arrastando
        vagas[vaga_idx][ponto_idx] = (x_original, y_original)

    elif event == cv2.EVENT_LBUTTONUP:
        ponto_arrastando = None

    elif event == cv2.EVENT_RBUTTONDOWN:
        salvar_vagas()


def desenhar_interface(img):
    img_display = img.copy()
    overlay = img_display.copy()
    vaga_hover = encontrar_vaga_sob_mouse(*mouse_pos)

    for idx, vaga in enumerate(vagas, start=1):
        vaga_poly = np.array(vaga, np.int32)
        selecionada = vaga_selecionada == idx - 1
        cor = (255, 255, 0) if selecionada else ((0, 180, 255) if vaga_hover == idx - 1 else (0, 255, 0))
        cv2.fillPoly(overlay, [vaga_poly], cor)
        cv2.polylines(img_display, [vaga_poly], True, cor, 3 if selecionada else 2)

        for px, py in vaga:
            cv2.circle(img_display, (px, py), 5, (0, 255, 255), -1)

        centro = np.mean(vaga_poly, axis=0).astype(int)
        cv2.putText(
            img_display,
            str(idx),
            tuple(centro),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    img_display = cv2.addWeighted(overlay, 0.18, img_display, 0.82, 0)

    if vaga_atual:
        for ponto in vaga_atual:
            cv2.circle(img_display, ponto, 5, (255, 0, 0), -1)
        if len(vaga_atual) > 1:
            cv2.polylines(img_display, [np.array(vaga_atual, np.int32)], False, (255, 0, 0), 2)
        cv2.line(img_display, vaga_atual[-1], mouse_pos, (255, 0, 0), 1)
        if len(vaga_atual) == 3:
            cv2.line(img_display, mouse_pos, vaga_atual[0], (255, 0, 0), 1)

    return img_display


vagas = carregar_vagas()
criar_imagem_base_pelo_video()
img = cv2.imread(str(IMAGEM_BASE))
if img is None:
    raise FileNotFoundError(f"Imagem base nao encontrada: {IMAGEM_BASE}")

escala_visualizacao = calcular_escala_visualizacao(img)
largura_exibida = int(img.shape[1] * escala_visualizacao)
altura_exibida = int(img.shape[0] * escala_visualizacao)
offset_x_visualizacao = (LARGURA_JANELA - largura_exibida) // 2
offset_y_visualizacao = (ALTURA_JANELA - altura_exibida) // 2
print(f"Janela do seletor em {LARGURA_JANELA}x{ALTURA_JANELA}. Coordenadas salvas no tamanho original.")

cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
cv2.resizeWindow(TITULO_JANELA, LARGURA_JANELA, ALTURA_JANELA)
cv2.setMouseCallback(TITULO_JANELA, mouse_click)

while True:
    frame_exibicao = preparar_para_exibir(desenhar_interface(img))
    cv2.imshow(TITULO_JANELA, desenhar_instrucoes(frame_exibicao))

    tecla = cv2.waitKeyEx(1)
    tecla_ascii = tecla & 0xFF
    if tecla_ascii in (ord('q'), ord('Q')):
        break
    if tecla_ascii in (ord('s'), ord('S')):
        salvar_vagas()
    elif tecla_ascii in (ord('u'), ord('U')):
        if vaga_atual:
            vaga_atual.pop()
        elif vagas:
            vagas.pop()
            if vaga_selecionada is not None and vaga_selecionada >= len(vagas):
                vaga_selecionada = len(vagas) - 1 if vagas else None
        print("Ultima acao desfeita.")
    elif tecla_ascii in (ord('d'), ord('D')):
        vaga_idx = vaga_selecionada if vaga_selecionada is not None else encontrar_vaga_sob_mouse(*mouse_pos)
        apagar_vaga(vaga_idx)
    elif tecla_ascii in (ord('c'), ord('C')):
        copiar_vaga_para_mouse()
    else:
        movimento = deslocamento_tecla(tecla)
        if movimento:
            mover_vaga(vaga_selecionada, *movimento)

cv2.destroyAllWindows()
