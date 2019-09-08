import cv2
import CycleGan
from torchvision.transforms import ToTensor, Resize
import argparse
from img_proc import process_image

parser = argparse.ArgumentParser()
parser.add_argument('-dir', type=str,
                    help='weight directory')
parser.add_argument('-e', '--epoch', type=int, help='epoch of the generated weights')
parser.add_argument('-d', default='cpu', type=str, help='device for the network')

args = vars(parser.parse_args())
print(args)

dir_weights = args['dir']

model = CycleGan.cycle_gan(training=False, device=args['d'])
model.load_weights(dir_weights, args['epoch'])

cap = cv2.VideoCapture(0)
r = Resize((1024, 720))
transformacion = ToTensor()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.resize(frame, (1024, 720))
    frame = transformacion(frame)
    frame = frame.unsqueeze(0)
    frame = frame.to('cpu')

    model.set_input_A(frame)
    model.forward_test()

    imagen = process_image(model.fake_B)

    # Display the resulting frame
    cv2.imshow('frame', imagen)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
