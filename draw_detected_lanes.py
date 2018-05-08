import numpy as np
import cv2
from scipy.misc import imresize
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from keras.models import load_model
from PIL import Image
from matplotlib import pyplot as plt
import pickle

# Load Keras model
model = load_model('CULane_CNN_model_shallow.h5')
# model = load_model('CULane_CNN_model_deep.h5')
# model = load_model('CULane_CNN_model_shallow.h5')

# Load training images
test_images = pickle.load(open("test_161_1.p", "rb"))

# Load image labels
labels = pickle.load(open("labels_161_1.p", "rb"))


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """

    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # show prediction
    # plt.imshow(lanes.avg_fit[:,:,0], interpolation='nearest')
    # plt.show()

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (720, 1280, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return result


# get squared error between prediction and ground truth label
def get_prediction_error(test_image, label_image, lanes):
    # Get image ready for feeding into model
    small_img = test_image[None, :, :, :]


    # Make prediction with neural network
    prediction = model.predict(small_img)[0]

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    avg_detection = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    max_pixel_value = np.max(avg_detection)

    avg_detection = avg_detection / max_pixel_value
    label_image = label_image / 255

    # show prediction
    # plt.imshow(lanes.avg_fit[:,:,0], interpolation='nearest')
    # plt.show()
    return np.sum((avg_detection - label_image) ** 2)


sum_error = 0
sum_pixel_error = 0
for i in range(len(test_images)):
    if i % 1000 == 0:
        print(i)
    lanes = Lanes()
    test_image = test_images[i]
    label_image = labels[i]
    # label test image
    error = get_prediction_error(test_image, label_image, lanes)
    sum_error += error
    sum_pixel_error += error / (test_image.shape[0] * test_image.shape[1])

print('Average sum of squared error =', sum_error / len(test_images))
print('Average sum of squared error (pixel level error) =', sum_pixel_error / len(test_images))

# lanes = Lanes()
#
# test_image = imresize(np.asarray(Image.open("sample_test.jpg")), (720, 1280, 3))
#
# # label test image
# img = road_lines(test_image)
#
# # show output
# plt.imshow(img, interpolation='nearest')
# plt.show()
#
# # save output
# plt.imshow(img, interpolation='nearest')
# plt.savefig('sample_output_deep.jpg')
