from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from CycleGan import cycle_gan

from img_proc import scale, save_image_tr


transform_test = transforms.Compose([transforms.Resize(720),  # puedes cambiar el tamaño de resize
                                    transforms.ToTensor()])


image_path_test_x = './input_imgs/'

test_x = ImageFolder(image_path_test_x, transform_test)

test_loader_x = DataLoader(dataset=test_x, batch_size=1, shuffle=False)

model = cycle_gan(training=False, device='cpu')  # Si tienes cuda cambiar 'cpu' -> 'cuda'
dir_weights = '../model_weights/'
# pesos disponibles para probar 60, 84, 100, 256, 280
model.load_weights(dir_weights, 40)

for num, batch in enumerate(test_loader_x):
    img, _ = batch
    model.set_input_A(scale(batch[0]))
    model.forward_test()

    save_image_tr(model.fake_B, num, 0, './', 'B')
