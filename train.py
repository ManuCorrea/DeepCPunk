import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


from CycleGan import cycle_gan

import numpy as np

import cv2

import datetime
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cuda'
print('###################  ', device, '    ################')

BATCH_SIZE = 5
RESUME_TRAINING = False


transform = transforms.Compose([transforms.RandomCrop(600, pad_if_needed=True),
                                transforms.Resize(512),
                                transforms.RandomHorizontalFlip(),
                                #transforms.Resize((720, 1024)),
                                transforms.ToTensor()])

transform_test = transforms.Compose([transforms.Resize((720, 1024)),
                                    transforms.ToTensor()])

# transform_test = transforms.Compose([transforms.ToTensor()])

image_path_train_x = './datasets/City/'
image_path_train_y = './datasets/CP/'
image_path_test_x = './datasets/test_{}/'.format('City')
image_path_test_y = './datasets/test_{}/'.format('CP')

train_x = ImageFolder(image_path_train_x, transform)
train_y = ImageFolder(image_path_train_y, transform)
test_x = ImageFolder(image_path_test_x, transform_test)
test_y = ImageFolder(image_path_test_y, transform_test)

train_loader_x = DataLoader(dataset=train_x, batch_size=BATCH_SIZE, shuffle=True)
test_loader_x = DataLoader(dataset=test_x, batch_size=1, shuffle=False)

train_loader_y = DataLoader(dataset=train_y, batch_size=BATCH_SIZE, shuffle=True)
test_loader_y = DataLoader(dataset=test_y, batch_size=1, shuffle=False)


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.
       This function assumes that the input x is already scaled from 0-1.'''

    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def save_image(img_tensor, img_num, epoch, direc, fake):
    img = img_tensor.squeeze(0)
    img = img.detach().cpu().numpy()
    # rgb bgr
    img = ((img + 1) * 255 / (2)).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    cv2.imwrite(direc + '/' + fake + '-num-{}epoch-{}.png'.format(img_num, epoch), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


dir_weights = './model_weights/512_nueva_red/'

modelo = cycle_gan()

if RESUME_TRAINING:
    modelo.load_weights(dir_weights, 200)

# number of epochs to train the model
n_epochs = 2000

started = datetime.datetime.now()
print('Started at:', started)

for epoch in range(102, n_epochs + 1):
    start_time = time.time()
    print('EPOCH NUM:', epoch)

    # Training
    for data_x, data_y in zip(train_loader_x, train_loader_y):
        
        modelo.set_input(scale(data_x[0]), scale(data_y[0]))
        modelo.optimize_parameters()

        # TODO monitor training losses properly
        # loss_G, loss_D_A, loss_D_B = modelo.get_losses()

    time_elapsed = time.time() - start_time
    print('Epoch {} took {:.2f} seconds ({:.2f} minutes)'.format(epoch, time_elapsed, time_elapsed/60))

    # testing
    if epoch % 2 == 0:
        modelo.print_losses()
        for num, batch in enumerate(test_loader_x):
            img, _ = batch
            modelo.set_input_test(scale(batch[0]))
            modelo.forward_test()

            save_image(modelo.fake_B, num, epoch, './samples/512/', 'B')
            save_image(modelo.fake_A, num, epoch, './samples/512/', 'A')

        modelo.save_weights(dir_weights, epoch)

print('Started at:', started)
print('Finished at ', datetime.datetime.now())
