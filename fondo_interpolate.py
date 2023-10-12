import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.nn import functional as F
import argparse

import torch

import time

start_time = time.time()
frame_count = 0


print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

args = (0, 1, 0.02, 8,'train_log')

try:
    try:
        try:
            from model.oldmodel.RIFE_HDv2 import Model
            model = Model()
            model.load_model(args[4], -1)
            print("Loaded v2.x HD model.")
        except:
            from train_log.RIFE_HDv3 import Model
            model = Model()
            model.load_model(args[4], -1)
            print("Loaded v3.x HD model.")
    except:
        from model.oldmodel.RIFE_HD import Model
        model = Model()
        model.load_model(args[4], -1)
        print("Loaded v1.x HD model")
except:
    from model.RIFE import Model
    model = Model()
    model.load_model(args[4], -1)
    print("Loaded ArXiv-RIFE model")

# Importar el modelo de interpolación
model.eval()
model.device()


cap_selfie_segmentation = mp.solutions.selfie_segmentation

# Nueva resolución deseada del cuadro de video
new_width = 640
new_height = 480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

BG_COLOR = (255, 0, 48)
IMAGE_PATH = "C:/Users/Andres_Elcientifico/Pictures/Captura de pantalla 2023-08-16 162730.png"
BG_IMAGE = cv2.imread(IMAGE_PATH)
BG_IMAGE = cv2.resize(BG_IMAGE, (new_width, new_height))  # Redimensionar imagen de fondo

font = cv2.FONT_HERSHEY_SIMPLEX  # Fuente para mostrar el FPS
font_scale = 1
font_color = (255, 255, 255)  # Color del texto

with cap_selfie_segmentation.SelfieSegmentation(
    model_selection = 0) as selfie_segmentation:
    
    while True:
        frame_count += 1
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)
        
        _, th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)
        th = th.astype(np.uint8)
        th = cv2.medianBlur(th, 19)
        th_inv = cv2.bitwise_not(th)

        # Background
        bg_image = np.ones(frame.shape, dtype=np.uint8)
        bg_image[:] = BG_IMAGE
        bg = cv2.bitwise_and(bg_image, bg_image, mask=th_inv)

        # Foreground
        fg = cv2.bitwise_and(frame, frame, mask=th)

        iteration = 0
        img0 = None
        img1 = None
        # Background + Foreground
        while iteration<2:
            output_cap = cv2.add(bg, fg)
            
            if img0 is None:
                img0 = output_cap
                # Transponer las dimensiones para que coincidan con la forma deseada (C, H, W)
                img0 = np.transpose(img0, (2, 0, 1))
                # Agregar una dimensión adicional para el lote
                img0 = np.expand_dims(img0, axis=0)
                # Convertir los arrays NumPy en tensores de PyTorch
                img0 = torch.tensor(img0, dtype=torch.float32)
                # Mover img0 e img1 a la GPU
                img0 = img0.to('cuda')
            elif iteration==1 and img1 is None:
                img1 = output_cap
                # Transponer las dimensiones para que coincidan con la forma deseada (C, H, W)
                img1 = np.transpose(img1, (2, 0, 1))
                # Agregar una dimensión adicional para el lote
                img1 = np.expand_dims(img1, axis=0)
                # Convertir los arrays NumPy en tensores de PyTorch
                img1 = torch.tensor(img1, dtype=torch.float32)
                # Mover img0 e img1 a la GPU
                img1 = img1.to('cuda')
            iteration += 1
       
        #cv2.imshow('frame', output_cap)

        if np.array_equal(img0, img1):
            print("No se detectó movimiento.")
        else:
            n, c, h, w = img0.shape
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            if args[1]:
                img_list = [img0]
                img0_ratio = 0.0
                img1_ratio = 1.0
                if args[1] <= img0_ratio + args[2] / 2:
                    middle = img0
                elif args[1] >= img1_ratio - args[2] / 2:
                    middle = img1
                else:
                    tmp_img0 = img0
                    tmp_img1 = img1
                    for inference_cycle in range(args[3]):
                        middle = model.inference(tmp_img0, tmp_img1)
                        middle_ratio = ( img0_ratio + img1_ratio ) / 2
                        if args[1] - (args[2] / 2) <= middle_ratio <= args[1] + (args[2] / 2):
                            break
                        if args[1] > middle_ratio:
                            tmp_img0 = middle
                            img0_ratio = middle_ratio
                        else:
                            tmp_img1 = middle
                            img1_ratio = middle_ratio
                img_list.append(middle)
                img_list.append(img1)
            else:
                img_list = [img0, img1]
                for i in range(args[0]):
                    tmp = []
                    for j in range(len(img_list) - 1):
                        mid = model.inference(img_list[j], img_list[j + 1])
                        tmp.append(img_list[j])
                        tmp.append(mid)
                    tmp.append(img1)
                    img_list = tmp
            img = img_list[0][0].cpu().numpy().transpose(1, 2, 0)[:h, :w]
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)

            if time.time() - start_time >= 1.0:
                fps = frame_count * len(img_list) / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), font, font_scale, font_color, 2)
            cv2.imshow('Imagen', img)

        

        if cv2.waitKey(1) & 0xff == 27:
            break
    
cap.release()
cv2.destroyAllWindows()