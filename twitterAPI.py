import tweepy
from APIkeys import *
from CycleGan import cycle_gan
from torchvision.transforms import ToTensor

from PIL import Image
import requests
from io import BytesIO

import numpy as np
import cv2

HASHTAG = 'DeepCPunk'  # Hashtag needed to process info


def process_image(img_tensor):
    img = img_tensor.squeeze(0)
    img = img.detach().cpu().numpy()
    # rgb bgr
    img = ((img + 1) * 255 / 2).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    return img

# resize and keep ratio
def resize_k_r(img, long_size):
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

transformation = ToTensor()

model = cycle_gan(training=False)
dir_weights = './model_weights/512_nueva_red/'
model.load_weights(dir_weights, 200)

# API initialization
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit_notify=True)
print(api.me())

mentions = api.mentions_timeline(tweet_mode='extended')

for tweet in mentions:
    tweet = tweet._json
    tweet_id = tweet['id']
    entities = tweet['entities']
    hashtags = entities['hashtags']

    if len(hashtags) > 0:
        # print(hashtags)
        for tag in hashtags:
            print(tag['text'])
        hashtags = [tag['text'] for tag in hashtags]

        if HASHTAG in hashtags:
            print(tweet)
            if 'extended_entities' in tweet.keys():
                extended_entities = tweet['extended_entities']  # If there is not media the tweet doesn't contains ext_ent
                media = extended_entities['media']  # Media is a list with dictionaries
                url = media[0]['media_url']
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img = resize_k_r(img, 1024)
                img = transformation(img)
                img = img.unsqueeze(0)
                # Forward pass of the image
                model.set_input_test(img)
                model.forward_test()

                # Save the image
                imagen = process_image(model.fake_B)
                img = Image.fromarray(imagen, 'RGB')
                img.save('./tw_imgs/{}.png'.format(10))
                cv2.imwrite('./tw_imgs/{}.png'.format(tweet_id),
                            cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR))

                api.update_with_media('./tw_imgs/{}.png'.format(tweet_id), '@{}'.format(tweet['user']['screen_name']),
                                      in_reply_to_status_id=tweet_id)
            else:
                print('no media')
    else:
        print('no hashtags')
