import cv2
import pickle
import numpy as np

vagas = []
vaga_atual = []

def mouse_click(events, x, y, flags, params):
    global vaga_atual
    if events == cv2.EVENT_LBUTTONDOWN:
        vaga_atual.append((x, y))
        if len(vaga_atual) == 4:
            vagas.append(vaga_atual)
            vaga_atual = []
    
    if events == cv2.EVENT_RBUTTONDOWN:
        with open('vagas_coordenadas.pkl', 'wb') as f:
            pickle.dump(vagas, f)
        print("Vagas salvas!")

img = cv2.imread('estacionamento_base.png')

while True:
    img_display = img.copy()
    for vaga in vagas:
        cv2.polylines(img_display, [np.array(vaga)], True, (0, 255, 0), 2)
    
    cv2.imshow("Clique em 4 pontos por vaga | Botao Direito p/ Salvar", img_display)
    cv2.setMouseCallback("Clique em 4 pontos por vaga | Botao Direito p/ Salvar", mouse_click)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()