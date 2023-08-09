import cv2
import mediapipe as mp
import numpy as np

cap_selfie_sepmentation = mp.solutions.selfie_segmentation

# Nueva resolución deseada del cuadro de video
new_width = 1920
new_height = 1080

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

BG_COLOR = (255, 0, 48)
IMAGE_PATH = "C:/Users/Andres_Elcientifico/Documents/miro.jpg"
BG_IMAGE = cv2.imread(IMAGE_PATH)
BG_IMAGE = cv2.resize(BG_IMAGE, (new_width, new_height))  # Redimensionar imagen de fondo


with cap_selfie_sepmentation.SelfieSegmentation(
    model_selection = 0) as selfie_segmentation:
    
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(frame_rgb)

        _, th = cv2.threshold(results.segmentation_mask, 0.75,255, cv2.THRESH_BINARY)
        th = th.astype(np.uint8)
        th = cv2.medianBlur(th,19)
        th_inv = cv2.bitwise_not(th)

        #Background
        bg_image = np.ones(frame.shape, dtype=np.uint8)
        bg_image[:] = BG_IMAGE
        bg = cv2.bitwise_and(bg_image, bg_image, mask=th_inv)

        #Foreground
        fg = cv2.bitwise_and(frame,frame, mask=th)

        #Background + Foreground
        output_cap = cv2.add(bg, fg)


        #cv2.imshow("results.segmentation_mask", results.segmentation_mask)
        #cv2.imshow("Th", th)
        #cv2.imshow("Th", th_inv)
        #cv2.imshow("BG_IMAGE", bg_image)
        #cv2.imshow("Frame", frame)
        #cv2.imshow("BG", bg)
        #cv2.imshow("FG", fg)
        cv2.imshow("Cap", output_cap)

        if cv2.waitKey(1)  & 0xff ==27:
            break

cap.realease()
cv2.destroyAllWindows()