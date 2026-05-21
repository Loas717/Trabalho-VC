import cv2
import pickle
import numpy as np
from pathlib import Path

ARQUIVO_COORDENADAS = Path('vagas_coordenadas.pkl')
IMAGEM_BASE = Path('estacionamento_base.png')
TITULO_JANELA = "Editor de vagas"
RAIO_SELECAO = 12

vagas = []
vaga_atual = []
ponto_arrastando = None
mouse_pos = (0, 0)


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


def mouse_click(event, x, y, flags, params):
    global vaga_atual, ponto_arrastando, mouse_pos
    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        ponto_proximo = encontrar_ponto_proximo(x, y)
        if ponto_proximo:
            ponto_arrastando = ponto_proximo
            return

        vaga_atual.append((x, y))
        if len(vaga_atual) == 4:
            vagas.append(vaga_atual.copy())
            vaga_atual = []
            print(f"Nova vaga adicionada. Total: {len(vagas)}")

    elif event == cv2.EVENT_MOUSEMOVE and ponto_arrastando:
        vaga_idx, ponto_idx = ponto_arrastando
        vagas[vaga_idx][ponto_idx] = (x, y)

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
        cor = (0, 180, 255) if vaga_hover == idx - 1 else (0, 255, 0)
        cv2.fillPoly(overlay, [vaga_poly], cor)
        cv2.polylines(img_display, [vaga_poly], True, cor, 2)

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

    instrucoes = [
        "Clique: novo ponto | Arraste ponto amarelo: ajustar recorte",
        "Botao direito ou S: salvar | D: apagar vaga sob mouse | U: desfazer | Q: sair",
        f"Vagas: {len(vagas)} | Pontos da nova vaga: {len(vaga_atual)}/4",
    ]

    y = 25
    for texto in instrucoes:
        cv2.putText(img_display, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img_display, texto, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25

    return img_display


vagas = carregar_vagas()
img = cv2.imread(str(IMAGEM_BASE))
if img is None:
    raise FileNotFoundError(f"Imagem base nao encontrada: {IMAGEM_BASE}")

cv2.namedWindow(TITULO_JANELA)
cv2.setMouseCallback(TITULO_JANELA, mouse_click)

while True:
    cv2.imshow(TITULO_JANELA, desenhar_interface(img))

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    if tecla == ord('s'):
        salvar_vagas()
    elif tecla == ord('u'):
        if vaga_atual:
            vaga_atual.pop()
        elif vagas:
            vagas.pop()
        print("Ultima acao desfeita.")
    elif tecla == ord('d'):
        vaga_idx = encontrar_vaga_sob_mouse(*mouse_pos)
        if vaga_idx is not None:
            vagas.pop(vaga_idx)
            print(f"Vaga {vaga_idx + 1} removida.")

cv2.destroyAllWindows()
