import numpy as np
import cv2


def process_image(img_tensor):
    """
    Takes a tensor and returns a numpy img.
    :param img_tensor:
    :return: image in numpy array
    """
    img = img_tensor.squeeze(0)
    img = img.detach().cpu().numpy()
    img = ((img + 1) * 255 / 2).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img


# resize and keep ratio
def resize_k_r(img, long_size):
    """
    Resize an img keeping the aspect ratio
    :param img: input img
    :param long_size: longest desired size of the ratio
    :return:
    """
    width, height = img.size
    ratio = width/height
    if width > height:
        if width < long_size:  # If the image is smaller don't resize
            return img
        print(long_size, int(long_size/ratio))
        return img.resize((long_size, int(long_size/ratio)))
    else:
        if height < long_size:
            return img
        print(int(ratio*long_size), long_size)
        return img.resize((int(ratio*long_size), long_size))


def scale(x, feature_range=(-1, 1)):
    """
    Scale takes in an image x and returns that image, scaled
    with a feature_range of pixel values from -1 to 1.
    This function assumes that the input x is already scaled from 0-1.
    """
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


def save_image_tr(img_tensor, img_num, epoch, direc, A_B):
    """
    Saves image and keeping track of what epoch it was
    :param img_tensor:
    :param img_num: Img number (identity)
    :param epoch: Epoch on which was generated
    :param direc: Directory to save it
    :param A_B: Type of image A or B
    :return:
    """
    img = img_tensor.squeeze(0)
    img = img.detach().cpu().numpy()
    img = ((img + 1) * 255 / 2).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    cv2.imwrite(direc + '/' + A_B + '-num-{}epoch-{}.png'.format(img_num, epoch), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # img.save(direc + '/' + A_B + '-num-{}epoch-{}.png'.format(img_num, epoch))

