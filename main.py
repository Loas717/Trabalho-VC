import cv2
import pickle
import numpy as np
from collections import deque
from ultralytics import YOLO

MODELO_YOLO = 'yolo11s.pt'
ARQUIVO_COORDENADAS = 'vagas_coordenadas.pkl'
VIDEO_ENTRADA = 'estacionamento_video.mp4'
CLASSES_VEICULOS = [2, 3, 5, 7]  # car, motorcycle, bus, truck no COCO
CONFIANCA_MINIMA = 0.30
IOU_YOLO = 0.50
LIMIAR_SOBREPOSICAO_VAGA = 0.12
JANELA_SUAVIZACAO = 5
MIN_OCUPACOES_NA_JANELA = 3

model = YOLO(MODELO_YOLO)

with open(ARQUIVO_COORDENADAS, 'rb') as f:
    vagas_pos = pickle.load(f)

video = cv2.VideoCapture(VIDEO_ENTRADA)
historico_vagas = [deque(maxlen=JANELA_SUAVIZACAO) for _ in vagas_pos]


def centro_da_caixa(caixa):
    x1, y1, x2, y2 = caixa
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def proporcao_sobreposta(vaga_poly, caixa, tamanho_frame):
    altura, largura = tamanho_frame[:2]
    x1, y1, x2, y2 = caixa
    x1 = max(0, min(largura - 1, x1))
    x2 = max(0, min(largura - 1, x2))
    y1 = max(0, min(altura - 1, y1))
    y2 = max(0, min(altura - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0

    mascara_vaga = np.zeros((altura, largura), dtype=np.uint8)
    mascara_caixa = np.zeros((altura, largura), dtype=np.uint8)

    cv2.fillPoly(mascara_vaga, [vaga_poly], 255)
    cv2.rectangle(mascara_caixa, (x1, y1), (x2, y2), 255, -1)

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

while True:
    check, frame = video.read()
    if not check: break

    results = model.predict(
        frame,
        classes=CLASSES_VEICULOS,
        conf=CONFIANCA_MINIMA,
        iou=IOU_YOLO,
        verbose=False
    )

    caixas_veiculos = []

    for r in results:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            caixas_veiculos.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"VEICULO {float(conf):.2f}", (x1, y1 - 10),
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

    cv2.imshow("Deteccao com YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
