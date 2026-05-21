import cv2
import pickle
import numpy as np
from collections import deque
from ultralytics import YOLO
from cenarios import escolher_cenario

MODELO_YOLO = 'yolo11s.pt'
CLASSES_VEICULOS = [2, 3, 5, 7]  # car, motorcycle, bus, truck no COCO
CONFIANCA_MINIMA = 0.30
IOU_YOLO = 0.50
LIMIAR_SOBREPOSICAO_VAGA = 0.12
JANELA_SUAVIZACAO = 5
MIN_OCUPACOES_NA_JANELA = 3
TITULO_JANELA = "Deteccao com YOLO"
LARGURA_JANELA = 1280
ALTURA_JANELA = 720
LARGURA_INFERENCIA = 960
PULAR_FRAMES = True
DETECTAR_A_CADA_N_FRAMES = 3

cena = escolher_cenario("rodar a deteccao")
ARQUIVO_COORDENADAS = cena.coordenadas
VIDEO_ENTRADA = cena.video

if not VIDEO_ENTRADA.exists():
    raise FileNotFoundError(f"Video do cenario nao encontrado: {VIDEO_ENTRADA}")

if not ARQUIVO_COORDENADAS.exists():
    raise FileNotFoundError(
        f"Coordenadas nao encontradas: {ARQUIVO_COORDENADAS}. Rode primeiro: python selector.py"
    )

model = YOLO(MODELO_YOLO)

with open(ARQUIVO_COORDENADAS, 'rb') as f:
    vagas_pos = pickle.load(f)

video = cv2.VideoCapture(VIDEO_ENTRADA)
historico_vagas = [deque(maxlen=JANELA_SUAVIZACAO) for _ in vagas_pos]


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


def preparar_para_inferencia(frame):
    altura, largura = frame.shape[:2]
    if largura <= LARGURA_INFERENCIA:
        return frame, 1.0, 1.0

    escala = LARGURA_INFERENCIA / largura
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


def centro_da_caixa(caixa):
    x1, y1, x2, y2 = caixa
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def proporcao_sobreposta(vaga_poly, caixa, tamanho_frame):
    altura_frame, largura_frame = tamanho_frame[:2]
    x1, y1, x2, y2 = caixa
    x1 = max(0, min(largura_frame - 1, x1))
    x2 = max(0, min(largura_frame - 1, x2))
    y1 = max(0, min(altura_frame - 1, y1))
    y2 = max(0, min(altura_frame - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0

    vx, vy, vw, vh = cv2.boundingRect(vaga_poly)
    rx1 = max(0, min(x1, vx))
    ry1 = max(0, min(y1, vy))
    rx2 = min(largura_frame - 1, max(x2, vx + vw))
    ry2 = min(altura_frame - 1, max(y2, vy + vh))

    largura_roi = rx2 - rx1
    altura_roi = ry2 - ry1
    if largura_roi <= 0 or altura_roi <= 0:
        return 0

    vaga_roi = vaga_poly - np.array([rx1, ry1])
    caixa_roi = (x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1)

    mascara_vaga = np.zeros((altura_roi, largura_roi), dtype=np.uint8)
    mascara_caixa = np.zeros((altura_roi, largura_roi), dtype=np.uint8)

    cv2.fillPoly(mascara_vaga, [vaga_roi], 255)
    cv2.rectangle(mascara_caixa, caixa_roi[:2], caixa_roi[2:], 255, -1)

    area_vaga = cv2.countNonZero(mascara_vaga)
    if area_vaga == 0:
        return 0

    area_intersecao = cv2.countNonZero(cv2.bitwise_and(mascara_vaga, mascara_caixa))
    return area_intersecao / area_vaga


def vaga_esta_ocupada(vaga_poly, caixas_veiculos, tamanho_frame):
    for caixa in caixas_veiculos:
        cx, cy = centro_da_caixa(caixa)
        centro_dentro = cv2.pointPolygonTest(vaga_poly, (cx, cy), False) >= 0
        sobreposicao = proporcao_sobreposta(vaga_poly, caixa, tamanho_frame)

        if centro_dentro or sobreposicao >= LIMIAR_SOBREPOSICAO_VAGA:
            return True

    return False

escala_visualizacao = None
fps_video = video.get(cv2.CAP_PROP_FPS) or 30
tempo_ultimo_frame = cv2.getTickCount()
numero_frame = 0
caixas_veiculos = []
deteccoes_atuais = []

while True:
    check, frame = video.read()
    if not check: break
    numero_frame += 1

    if escala_visualizacao is None:
        escala_visualizacao = calcular_escala_visualizacao(frame)
        cv2.namedWindow(TITULO_JANELA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITULO_JANELA, LARGURA_JANELA, ALTURA_JANELA)
        print(f"Janela da deteccao em {LARGURA_JANELA}x{ALTURA_JANELA}. Processamento no tamanho original.")

    if numero_frame == 1 or numero_frame % DETECTAR_A_CADA_N_FRAMES == 0:
        frame_inferencia, escala_x_inferencia, escala_y_inferencia = preparar_para_inferencia(frame)

        results = model.predict(
            frame_inferencia,
            classes=CLASSES_VEICULOS,
            conf=CONFIANCA_MINIMA,
            iou=IOU_YOLO,
            verbose=False
        )

        caixas_veiculos = []
        deteccoes_atuais = []

        for r in results:
            for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
                caixa_original = converter_caixa_para_original(
                    tuple(map(int, box)),
                    escala_x_inferencia,
                    escala_y_inferencia
                )
                caixas_veiculos.append(caixa_original)
                deteccoes_atuais.append((caixa_original, float(conf)))

    for caixa_original, conf in deteccoes_atuais:
        x1, y1, x2, y2 = caixa_original
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"VEICULO {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    vagas_livres = 0
    vagas_ocupadas = 0

    for idx, vaga in enumerate(vagas_pos):
        vaga_poly = np.array(vaga, np.int32)
        ocupada_agora = vaga_esta_ocupada(vaga_poly, caixas_veiculos, frame.shape)
        historico_vagas[idx].append(ocupada_agora)
        ocupada = sum(historico_vagas[idx]) >= MIN_OCUPACOES_NA_JANELA
        
        color = (0, 0, 255) if ocupada else (0, 255, 0)
        if ocupada:
            vagas_ocupadas += 1
        else:
            vagas_livres += 1
        
        cv2.polylines(frame, [vaga_poly], True, color, 2)

    cv2.putText(frame, f'Vagas Livres: {vagas_livres}', (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(frame, f'Vagas Ocupadas: {vagas_ocupadas}', (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow(TITULO_JANELA, preparar_para_exibir(frame, escala_visualizacao))

    tempo_frame_ms = 1000 / fps_video
    tempo_agora = cv2.getTickCount()
    tempo_processado_ms = ((tempo_agora - tempo_ultimo_frame) / cv2.getTickFrequency()) * 1000
    tempo_ultimo_frame = tempo_agora
    espera_ms = max(1, int(tempo_frame_ms - tempo_processado_ms))

    if PULAR_FRAMES and tempo_processado_ms > tempo_frame_ms:
        frames_para_pular = int(tempo_processado_ms // tempo_frame_ms) - 1
        for _ in range(max(0, frames_para_pular)):
            video.grab()

    if cv2.waitKey(espera_ms) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
