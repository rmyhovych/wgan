import os
import torch
import random

from mnist.loader import MNIST

DIR = os.path.dirname(os.path.realpath(__file__))

PATH_RAW = os.path.join(DIR, "raw")
PATH_IMAGES = os.path.join(DIR, "images.tensor")
PATH_LABELS = os.path.join(DIR, "labels.tensor")


def save():
    mndata = MNIST(PATH_RAW)

    images, labels = mndata.load_training()
    data = list(zip(images, labels))
    random.shuffle(data)

    images = [d[0] for d in data]
    labels = [d[1] for d in data]

    images = torch.tensor(
        [[float(j) / 256.0 for j in i] for i in images], dtype=torch.float
    )
    labels = torch.tensor([[j] for j in labels], dtype=torch.long)

    torch.save(images, PATH_IMAGES)
    torch.save(labels, PATH_LABELS)


def load():
    return torch.load(PATH_IMAGES), torch.load(PATH_LABELS)


if __name__ == "__main__":
    save()
