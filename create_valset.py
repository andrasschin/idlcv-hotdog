import os
from glob import glob
import random
import sys

random.seed(0)

base_path = "/work3/s233084/datasets/hotdog_nothotdog/"

if os.path.exists(base_path + "validation"):
    sys.exit(0)
else:
    os.mkdir(base_path + "validation")
    os.mkdir(base_path + "validation/hotdog")
    os.mkdir(base_path + "validation/nothotdog")

hotdogs = glob(base_path + "train/hotdog/*.jpg")
nothotdogs = glob(base_path + "train/nothotdog/*.jpg")

n_hotdogs = int(len(hotdogs) * 0.2)
n_nothotdogs = int(len(nothotdogs) * 0.2)

val_hotdogs = random.sample(hotdogs, n_hotdogs)
val_nothotdogs = random.sample(nothotdogs, n_nothotdogs)

for fname in val_hotdogs:
    os.rename(fname, fname.replace("train", "validation"))

for fname in val_nothotdogs:
    os.rename(fname, fname.replace("train", "validation"))


train_hotdogs = glob(base_path + "train/hotdog/*.jpg")
train_nothotdogs = glob(base_path + "train/nothotdog/*.jpg")
val_hotdogs = glob(base_path + "validation/hotdog/*.jpg")
val_nothotdogs = glob(base_path + "validation/nothotdog/*.jpg")
print(len(train_hotdogs), len(train_nothotdogs))
print(len(val_hotdogs), len(val_nothotdogs))
