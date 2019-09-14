import cv2
import numpy as np
import CycleGan
from torchvision.transforms import ToTensor
import argparse
from img_proc import process_image

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str, default='../model_weights/',
                    help='weight directory')
parser.add_argument('-e', '--epoch', default=100,type=int, help='epoch of the generated weights')
parser.add_argument('-d', default='cuda', type=str, help='device for the network')

args = vars(parser.parse_args())
print(args)

dir_weights = args['dir']

model = CycleGan.cycle_gan(training=False, device=args['d'])
model.load_weights(dir_weights, args['epoch'])



file_read = 'granada.webm'
cap = cv2.VideoCapture(file_read)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print(frame_width, frame_height)
out = cv2.VideoWriter('granada-BGR.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
                      10, (frame_width,frame_height))

transformacion = ToTensor()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # frame = cv2.resize(frame, (2200, 917))
        frame = transformacion(frame)
        frame = frame.unsqueeze(0)


        model.set_input_A(frame)
        model.forward_AB()

        imagen = process_image(model.fake_B)
        imagen = process_image(model.fake_B)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        out.write(imagen)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
