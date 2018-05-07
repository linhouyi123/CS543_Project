import pickle
import csv
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def downsample(img, w, h):
    if 'P' in img.mode:  # check if image is a palette type
        img = img.convert("RGB")  # convert it to RGB
        img = img.resize((w, h), Image.ANTIALIAS)  # resize it
        img = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
        # convert back to palette
    else:
        img = img.resize((w, h), Image.ANTIALIAS)  # regular resize
    return img


with open('train_182.txt', newline='') as csvfile:
    batch_size = 20000
    train_images = []
    labels = []
    directory_reader = csv.reader(csvfile, delimiter=' ')
    i = 0
    num = 0
    for row in directory_reader:
        input_img_path = '.' + row[0]
        label_path = '.' + row[1]

        input_img = np.asarray(downsample(Image.open(input_img_path), 160, 80))
        label_img = np.asarray(downsample(Image.open(label_path), 160, 80))

        label_img.setflags(write=1)
        label_img[label_img != 0] = 255
        label_img = label_img[:, :, np.newaxis]
        # plt.imshow(label_img)
        # plt.show()

        train_images.append(input_img)
        labels.append(label_img)

        i += 1
        if i % 1000 == 0:
            print('i =', i, 'num =', num)
        if i >= batch_size:
            pickle.dump(train_images, open('train_182_' + str(num) + '.p', 'wb'))
            pickle.dump(labels, open('labels_182_' + str(num) + '.p', 'wb'))
            num += 1
            i = 0
            train_images.clear()
            labels.clear()

    if len(train_images) > 0:
        num += 1
        pickle.dump(train_images, open('train_182_' + str(num) + '.p', 'wb'))
        pickle.dump(labels, open('labels_182_' + str(num) + '.p', 'wb'))


