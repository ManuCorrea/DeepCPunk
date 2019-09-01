import cv2
import numpy as np
import CycleGan
from torchvision.transforms import ToTensor, Resize

def process_image(img_tensor):
    img = img_tensor.squeeze(0)
    img = img.detach().cpu().numpy()
    # rgb bgr
    img = ((img + 1) * 255 / (2)).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

dir_weights = './model_weights/512_nueva_red/'

model = CycleGan.cycle_gan(training=False)
model.load_weights(dir_weights, 424)

cap = cv2.VideoCapture(0)
r = Resize((1024, 720))
transformacion = ToTensor()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (1024, 720))
    frame = transformacion(frame)
    frame = frame.unsqueeze(0)
    frame = frame.to('cpu')

    model.set_input_test(frame)
    model.forward_test()

    imagen = process_image(model.fake_B)

    # Display the resulting frame
    cv2.imshow('frame',imagen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()