import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from CycleGan import cycle_gan

import matplotlib.pyplot as plt

import datetime
import time
import argparse
from img_proc import scale, save_image_tr


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-bs', default=5, type=int,
                    help='Batch Size for training')

parser.add_argument('--resume_training', default=False, type=bool,
                    help='resume training with previous saved weights')

parser.add_argument('-e', '--epoch', type=int, default=10,
                    help='Number of epochs')

parser.add_argument('--EpResTr', default=0, type=int,
                    help='Epoch number saved weight for training')

parser.add_argument('--dir', default='./model_weights/', type=str,
                    help='directory of weights')

parser.add_argument('--test_dir', default='./samples/', type=str,
                    help='directory for saving tests')

parser.add_argument('--each', type=int, default=4,
                    help='Number of epochs to test and save.')

args = vars(parser.parse_args())
print(args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('###################  ', device, '    ################')

BATCH_SIZE = args['bs']
RESUME_TRAINING = args['resume_training']


transform = transforms.Compose([transforms.RandomCrop(580, pad_if_needed=True),
                                transforms.Resize(512),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()])

transform_test = transforms.Compose([transforms.Resize(720),
                                    transforms.ToTensor()])

# transform_test = transforms.Compose([transforms.ToTensor()])
data_folder = './datasets/'

image_path_train_x = data_folder + 'City/City/'
image_path_train_y = data_folder + 'CP/CP/'
image_path_test_x = data_folder + 'test_{}/'.format('City')
image_path_test_y = data_folder + 'test_{}/'.format('CP')

train_x = ImageFolder(image_path_train_x, transform)
train_y = ImageFolder(image_path_train_y, transform)
test_x = ImageFolder(image_path_test_x, transform_test)
test_y = ImageFolder(image_path_test_y, transform_test)

train_loader_x = DataLoader(dataset=train_x, batch_size=BATCH_SIZE, shuffle=True)
test_loader_x = DataLoader(dataset=test_x, batch_size=1, shuffle=False)

train_loader_y = DataLoader(dataset=train_y, batch_size=BATCH_SIZE, shuffle=True)
test_loader_y = DataLoader(dataset=test_y, batch_size=1, shuffle=False)


dir_weights = args['dir']

modelo = cycle_gan(device)

if RESUME_TRAINING:
    modelo.load_weights(dir_weights, args['EpResTr'])

# number of epochs to train the model
n_epochs = args['epoch']

started = datetime.datetime.now()
print('Started at:', started)

# Arrays to keep track of loss during training
loss_G_epoch = []
loss_DA_epoch = []
loss_DB_epoch = []
loss_G_epochs = []
loss_DA_epochs = []
loss_DB_epochs = []

for epoch in range(args['EpResTr'], n_epochs + 1):
    start_time = time.time()
    print('EPOCH NUM:', epoch)

    # Training
    for data_x, data_y in zip(train_loader_x, train_loader_y):
        modelo.set_input(scale(data_x[0]), scale(data_y[0]))
        modelo.optimize_parameters()

        loss_G, loss_DA, loss_DB = modelo.get_losses()
        loss_G_epoch.append(loss_G)
        loss_DA_epoch.append(loss_DA)
        loss_DB_epoch.append(loss_DB)

    loss_G_epochs.append(sum(loss_G_epoch)/len(loss_G_epoch))
    loss_DA_epochs.append(sum(loss_DA_epoch)/len(loss_DA_epoch))
    loss_DB_epochs.append(sum(loss_DB_epoch)/len(loss_DB_epoch))
    loss_G_epoch = []
    loss_DA_epoch = []
    loss_DB_epoch = []

    time_elapsed = time.time() - start_time
    print('Epoch {} took {:.2f} seconds ({:.2f} minutes)'.format(epoch, time_elapsed, time_elapsed/60))

    # testing
    if epoch % args['each'] == 0:
        for num, batch in enumerate(test_loader_x):
            img, _ = batch
            modelo.set_input_A(scale(batch[0]))
            modelo.forward_test()

            save_image_tr(modelo.fake_B, num, epoch, args['test_dir'], 'B')
            save_image_tr(modelo.fake_A, num, epoch, args['test_dir'], 'A')

        modelo.save_weights(dir_weights, epoch)

print('Started at:', started)
print('Finished at ', datetime.datetime.now())

plt.plot(loss_G_epochs, label='Generator losses')
plt.plot(loss_DA_epochs, label='Discriminator A losses')
plt.plot(loss_DB_epochs, label='Discriminator B losses')
plt.xlabel('Epochs')
plt.ylabel('Loss value')
plt.title('Training losses')
plt.legend()
plt.show()
