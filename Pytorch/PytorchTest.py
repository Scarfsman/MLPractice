# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:04:03 2023

@author: Georg
"""

import torch
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models

url = 'https://pytorch.tips/coffee'
fpath = 'coffee.jpg'
urllib.request.urlretrieve(url, fpath)
img = Image.open('coffee.jpg')
plt.imshow(img)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225])])

img_tensor = transform(img)

batch = img_tensor.unsqueeze(0)

model = models.alexnet(pretrained = True)
model.eval()
model.to('cpu')
y = model(batch.to('cpu'))

ymax, index = torch.max(y, 1)

url = 'https://pytorch.tips/imagenet-labels'

fpath = 'imagenet_class_labels'
urllib.request.urlretrieve(url, fpath)

with open(fpath) as f:
    classes = [line.strip() for line in f.readlines()]
    
prob = torch.nn.functional.softmax(y, dim = 1)[0] * 100
print(classes[index[0]], prob[index[0]].item())