# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ALLDataset.py
# 2023/05/10 to-arai

import os
import sys

import numpy as np

from tqdm import tqdm
import glob
# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label
from skimage.io import imread, imshow
from matplotlib import pyplot as plt
import traceback

class ALLDataset:

  def __init__(self, resized_image):
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = resized_image

 
  def create(self, original_data_path, segmented_data_path, category):
    image_files = glob.glob(original_data_path  + "/" + category + "/*.jpg")
    mask_files  = glob.glob(segmented_data_path + "/" + category + "/*.jpg")
    if len(image_files) != len(mask_files):
      raise Exception("The number of the original and segmented image files is not matched")
    num_images  = len(image_files)
    X = []
    Y = []

    X = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
    Y = np.zeros((num_images, self.IMG_HEIGHT, self.IMG_WIDTH, 1                ), dtype=bool)

    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
    
      basename  = os.path.basename(image_file)
      mask_file = segmented_data_path + "/" + category + "/" + basename

      image = imread(image_file)
      image = resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), mode='constant', preserve_range=True)
      X[n]  = image

      mask  = imread(mask_file)
      mask  = resize(mask, (self.IMG_HEIGHT, self.IMG_WIDTH, 1), mode='wrap', preserve_range=False, anti_aliasing=True)
      #imshow(mask)
      #plt.show()
      #input("XX")
      Y[n]  = mask

    return X, Y


if __name__ == "__main__":
  try:
    resized_image = (256, 256, 3)
    dataset = ALLDataset(resized_image)

    # train dataset
    original_data_path  = "./ALL/train/original/"
    segmented_data_path = "./ALL/train/segmented/"
    category            = "Early"
    x_train, y_train = dataset.create(original_data_path, segmented_data_path, category)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    original_data_path  = "./ALL/test/original/"
    segmented_data_path = "./ALL/test/segmented/"
    category            = "Early"

    x_test, y_test = dataset.create(original_data_path, segmented_data_path, category)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

