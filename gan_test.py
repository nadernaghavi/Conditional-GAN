# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 02:51:59 2019

@author: NaderBrave
"""

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cv2
from os import listdir
import os
import numpy as np
import collections
from keras.models import model_from_json

json_file = open('gan.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
gan = model_from_json(loaded_model_json)
# load weights into new model
gan.load_weights("gan.h5")

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")

print("Loaded model from disk")





ix2vocab = np.load('ix2vocab.npy').item()
vocab2ix = np.load('vocab2ix.npy').item()

text=['A','B','C',' ','D','E','F','G',' ','H','B','M','K','S']
text =['T','H','E',' ','Q','U','I','C','K',' ','B','R','O','W','N',' ','F','O','X',' ','J','U','M','P','S',' ',
       'O','V','E','R',' ','A',' ','L','A','Z','Y',' ','D','O','G']
#text = ['A','B','C','D','E','F','G','H']
sent=[]
for t in text:
    if t == ' ':
        temp = 0
        sent.append(0)
    else:
        temp = vocab2ix[t]
        sent.append(temp)
    


sent = np.array(sent)
    
image_sen=[]
font = 4121

for l in sent:
    if l ==0 :
        temp = np.ones((32,32))
        image_sen.append(temp)
    else:
        l = l -1 
        label1 = l*np.ones(10).reshape(-1,1)
        label2 = font*np.ones(10).reshape(-1,1)
        noise = np.random.normal(0,1,(10,100))
        imgs = gan.predict([noise, label1, label2])
        imgs = np.reshape(imgs,(10,32,32,1))
        scores = model.predict(imgs)
        scores = scores[:,l]
        best_arg = np.argmax(scores)
        img = imgs[best_arg]
        img = np.mean(img, axis=2)
        img = img*0.5 + 0.5
        image_sen.append(img)

image_sen = np.array(image_sen)
shape = image_sen.shape
image_sen1 = np.reshape(image_sen,(shape[0]*shape[1],shape[2]))
#image_sen = np.reshape(image_sen,(shape[1],shape[0]*shape[2]))
sample_image = image_sen1[0:32,:]
plt.show()
plt.imshow(image_sen1,cmap='gray')

final_image = np.zeros((shape[1],shape[0]*shape[2]))

#final_image = []

for i in range(shape[0]):
    final_image[:,i*32:(i+1)*32] = image_sen[i,:,:]
plt.show()
plt.imshow(final_image, cmap='gray')
final_image = 255*final_image
#final_image = final_image.astype('unit8')
cv2.imwrite("sentence.jpg",final_image)


               
   
