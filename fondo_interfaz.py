import time
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

cap_selfie_segmentation = mp.solutions.selfie_segmentation

new_width = 640
new_height = 480
BG_IMAGE = None
blur_value = 13
model_selection = 0

# Función para cambiar el fondo
def change_background():
    global BG_IMAGE
    file_path = filedialog.askopenfilename()
    if file_path:
        BG_IMAGE = cv2.imread(file_path)
        if BG_IMAGE is not None:
            BG_IMAGE = cv2.resize(BG_IMAGE, (new_width, new_height))
        else:
            print("Error: No se pudo cargar la imagen de fondo.")

# Función para actualizar la resolución
def update_resolution():
    global new_width, new_height
    new_width = int(resolution_width_entry.get())
    new_height = int(resolution_height_entry.get())
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    if BG_IMAGE is not None:
        BG_IMAGE = cv2.resize(BG_IMAGE, (new_width, new_height))

# Función para actualizar el desenfoque
def update_blur():
    global blur_value
    blur_value = int(blur_scale.get())

# Función para cambiar el modelo de segmentación
def change_model():
    global model_selection
    model_selection = model_var.get()

# Función para salir del bucle
def exit_loop():
    global should_exit
    should_exit = True

# Configuración de la interfaz
root = tk.Tk()
root.title("Besthelp Selfie Segmentation App")

should_exit = False  # Bandera para controlar la salida del bucle

# Botón para cambiar el fondo
bg_button = tk.Button(root, text="Cambiar Fondo", command=change_background)
bg_button.pack()

# Entradas para la resolución
resolution_width_label = tk.Label(root, text="Ancho:")
resolution_width_label.pack()
resolution_width_entry = tk.Entry(root)
resolution_width_entry.pack()

resolution_height_label = tk.Label(root, text="Alto:")
resolution_height_label.pack()
resolution_height_entry = tk.Entry(root)
resolution_height_entry.pack()

update_resolution_button = tk.Button(root, text="Actualizar Resolución", command=update_resolution)
update_resolution_button.pack()

# Control de desenfoque
blur_label = tk.Label(root, text="Desenfoque:")
blur_label.pack()
blur_scale = tk.Scale(root, from_=1, to=19, orient="horizontal")
blur_scale.pack()

update_blur_button = tk.Button(root, text="Actualizar Suavisado", command=update_blur)
update_blur_button.pack()

# Selección del modelo de segmentación
model_var = tk.IntVar()
model_var.set(model_selection)  # Valor predeterminado: Modelo 0
model_0_radio = tk.Radiobutton(root, text="Modelo bajo", variable=model_var, value=0, command=change_model)
model_0_radio.pack()

model_1_radio = tk.Radiobutton(root, text="Modelo medio", variable=model_var, value=1, command=change_model)
model_1_radio.pack()

# Configuración de la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

# Mostrar el primer marco para inicializar la ventana
ret, frame = cap.read()
if ret:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with cap_selfie_segmentation.SelfieSegmentation(model_selection=model_var.get()) as selfie_segmentation:
        results = selfie_segmentation.process(frame_rgb)

    # Procesar la imagen y mostrarla en la interfaz
    th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)[1]
    th = th.astype(np.uint8)
    th = cv2.medianBlur(th, blur_value)
    th_inv = cv2.bitwise_not(th)

    bg_image = np.ones(frame.shape, dtype=np.uint8)
    if BG_IMAGE is not None:
        bg_image = cv2.resize(BG_IMAGE, (new_width, new_height))
        bg = cv2.bitwise_and(bg_image, bg_image, mask=th_inv)
        fg = cv2.bitwise_and(frame, frame, mask=th)
        output_cap = cv2.addWeighted(bg, 1, fg, 1, 0)
    else:
        fg = cv2.bitwise_and(frame, frame, mask=th)
        output_cap = fg

    output_image = cv2.cvtColor(output_cap, cv2.COLOR_BGR2RGB)
    output_image_pil = Image.fromarray(output_image)
    output_image_tk = ImageTk.PhotoImage(output_image_pil)

    panel = tk.Label(root, image=output_image_tk)
    panel.image = output_image_tk
    panel.pack()

    root.update_idletasks()

# Iniciar el bucle principal
while not should_exit:
    ret, frame = cap.read()
    if ret == False:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with cap_selfie_segmentation.SelfieSegmentation(model_selection=model_var.get()) as selfie_segmentation:
        results = selfie_segmentation.process(image=frame_rgb)

    # Procesar la imagen y mostrarla en la interfaz
    th = cv2.threshold(results.segmentation_mask, 0.75, 255, cv2.THRESH_BINARY)[1]
    th = th.astype(np.uint8)
    th = cv2.medianBlur(th, blur_value)
    th_inv = cv2.bitwise_not(th)

    bg_image = np.ones(frame.shape, dtype=np.uint8)
    if BG_IMAGE is not None:
        bg_image = cv2.resize(BG_IMAGE, (new_width, new_height))
        bg = cv2.bitwise_and(bg_image, bg_image, mask=th_inv)
        fg = cv2.bitwise_and(frame, frame, mask=th)
        output_cap = cv2.addWeighted(bg, 1, fg, 1, 0)
    else:
        fg = cv2.bitwise_and(frame, frame, mask=th)
        output_cap = fg

    # Mostrar la imagen en una ventana de Pillow
    output_image = cv2.cvtColor(output_cap, cv2.COLOR_BGR2RGB)
    output_image_pil = Image.fromarray(output_image)
    output_image_tk = ImageTk.PhotoImage(output_image_pil)

    panel.configure(image=output_image_tk)  # Actualizar la imagen en la etiqueta
    root.update()  # Actualizar la ventana de Tkinter

    time.sleep(0.03)  # Pausa durante aproximadamente 30 milisegundos

# Cerrar la ventana y liberar recursos
cap.release()
cv2.destroyAllWindows()
root.destroy()