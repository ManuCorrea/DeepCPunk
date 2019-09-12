import argparse
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='resize all dataset and save it')

parser.add_argument('-s', default=680, type=int, help='Size to resize')

valid_images = [".jpg", ".png", 'jpeg']

args = vars(parser.parse_args())
print(args)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


SIZE = args['s']
print('Image are going to be resized with a size of: ', SIZE)
resize = transforms.Resize(SIZE)

city = './City/City/'
cp = './CP/CP/'
dest_city_dir = city + 'resized/'
dest_cp_dir = cp + 'resized/'

check_dir(dest_city_dir)
check_dir(dest_cp_dir)

for file in tqdm(os.listdir(city)):
    if os.path.splitext(file)[1].lower() in valid_images:
        img = Image.open(city + file)
        img = resize(img)
        img.save(dest_city_dir + file)

for file in tqdm(os.listdir(cp)):
    if os.path.splitext(file)[1].lower() in valid_images:
        img = Image.open(cp + file)
        img = resize(img)
        img.save(dest_cp_dir + file)
