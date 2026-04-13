import cv2
import pickle
import numpy as np
from ultralytics import YOLO

model = YOLO('yolo11n.pt') 

with open('vagas_coordenadas.pkl', 'rb') as f:
    vagas_pos = pickle.load(f)

video = cv2.VideoCapture('estacionamento_video.mp4')

while True:
    check, frame = video.read()
    if not check: break

    results = model.predict(frame, classes=[2, 5, 7], conf=0.4, iou=0.5, verbose=False)
    
    centros_veiculos = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            cv2.putText(frame, "CARRO", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)
            centros_veiculos.append((centro_x, centro_y))

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box
            centro_x = int((x1 + x2) / 2)
            centro_y = int((y1 + y2) / 2)
            centros_veiculos.append((centro_x, centro_y))

    vagas_livres = 0
    for vaga in vagas_pos:
        vaga_poly = np.array(vaga, np.int32)
        ocupada = False
        
        for cx, cy in centros_veiculos:
            dist = cv2.pointPolygonTest(vaga_poly, (cx, cy), False)
            if dist >= 0:
                ocupada = True
                break
        
        color = (0, 0, 255) if ocupada else (0, 255, 0)
        if not ocupada: vagas_livres += 1
        
        cv2.polylines(frame, [vaga_poly], True, color, 2)

    cv2.putText(frame, f'Vagas Livres: {vagas_livres}', (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Deteccao com YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break