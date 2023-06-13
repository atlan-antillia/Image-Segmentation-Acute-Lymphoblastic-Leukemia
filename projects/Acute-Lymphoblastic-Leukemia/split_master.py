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

# 2023/05/10 to-arai

# split_master.py
# This splits "Early master dataset in the following directory to train and test dataset.
#     MASTER_DIR          = "./Acute-Lymphoblastic-Leukemia/"
#    category            = "Early"
#
# The following dir tree will be generated.

"""
ALL
├─test
│  ├─original
│  │  └─Early
│  └─segmented
│      └─Early
└─train
    ├─original
    │  └─Early
    └─segmented
        └─Early
"""

import os
import shutil
import glob
import random
import traceback

if __name__ == "__main__":
  try:
    MASTER_DIR          = "./Acute-Lymphoblastic-Leukemia/"
    category            = "Early"

    master_original_dir = MASTER_DIR + "/Original/" + category

    image_files = glob.glob(master_original_dir  + "/*.jpg")

    num_images  = len(image_files)
    num_train   = int(num_images * 0.8)
    num_test    = int(num_images * 0.2)
    print("=== num_files {}".format(num_images))
    print("=== num_train {}".format(num_train))
    print("=== num_test  {}".format(num_test))

    random.shuffle(image_files)

    train_files = image_files[0:num_train]
    test_files  = image_files[num_train: num_images]

    train_test_files = [train_files, test_files]
     
    # 2 Copy the splitted train and test image files to train and test folder 
    origingal_train_test_dirs  = ["./ALL/train/original/", "./ALL/test/original/" ]

    for i, dir in enumerate(origingal_train_test_dirs):
       category_dir = os.path.join(dir, category)
       if os.path.exists(category_dir):
         shutil.rmtree(category_dir)
       if not os.path.exists(category_dir):
         os.makedirs(category_dir)

       files = train_test_files[i]
       for file in files:
         shutil.copy2(file, category_dir)
         print("=== Copied {} to {}".format(file, category_dir))
    master_segmented_dir       = MASTER_DIR + "/Segmented/" + category

    # 3 Copy the splitted train and test segmented image files to train and test folder 

    segmented_train_test_dirs  = ["./ALL/train/segmented/", "./ALL/test/segmented/" ]

    for i, dir in enumerate(segmented_train_test_dirs):
       category_dir = os.path.join(dir, category)
       if os.path.exists(category_dir):
         shutil.rmtree(category_dir)
       if not os.path.exists(category_dir):
         os.makedirs(category_dir)

       files = train_test_files[i]
       for file in files:
         basename = os.path.basename(file)
         segmented_file = os.path.join(master_segmented_dir, basename)
         shutil.copy2(segmented_file, category_dir)
         print("=== Copied {} to {}".format(segmented_file, category_dir))

  except:
    traceback.print_exc()
