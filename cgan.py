# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 08:50:27 2019

@author: NaderBrave
"""






from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
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
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
from keras import backend as K
from keras.utils.vis_utils import plot_model
K.set_image_dim_ordering('th')



data = np.load('full_data.npy')
data_label = np.load('full_label.npy')
data_label = data_label.reshape(-1,1)
#data = np.mean(data,axis=3)
#data = np.squeeze(data)
num_classes = 27
num_fonts = 9001
batch_size = 512
epochs = 10

font_label=[]
for m in range(9):
    for n in range(26):
        for i in range(1000):
            temp = 1000*m+i
            font_label.append(temp)
font_label = np.array(font_label)
font_label = font_label.reshape(-1,1)

data = data.reshape(len(data),1,32,32)
#data = np.squeeze(data)
length = len(data)
data1 = data[0].squeeze()
"""
font_label = []
length = len(data)
for i in range(length):
    temp = int(i/20)
    temp = i - 20*temp
    font_label.append(temp)

font_label = np.array(font_label)
font_label = font_label.reshape(-1,1)


data = data[0:26000]
data_label = data_label[0:26000]
font_label = font_label[0:26000]

data_temp=[]
label_temp = []
font_temp = []
for i in range(26):
    for j in range(20):
        rand = i*1000+j
        data_temp.append(data[rand])
        label_temp.append(data_label[rand])
        font_temp.append(font_label[rand])
        
data = np.array(data_temp)
data_label = np.array(label_temp)
font_label = np.array(font_temp)        

data_label = data_label.reshape(-1,1)      


def data_aug(data1,label1,count):
    data1 = data1.reshape(data1.shape[0],1,32,32)
    datagen = ImageDataGenerator()
    datagen.fit(data1)
    data = []
    data_label = []
    cnt = 0
    for x, y in datagen.flow(data1, label1, batch_size=1):
      data.append(x)
      data_label.append(y)
      cnt =  cnt + 1
      if cnt==count:
        break
    data = np.array(data)
    data_label = np.array(data_label)
    data = data.reshape(data.shape[0],1,32,32)
    data_label = np.squeeze(data_label)
    #data_label = data_label.reshape(-1,1)
    return data, data_label

label = np.concatenate([data_label,font_label],axis=1)

data,label = data_aug(data,label,100000)

data_label = label[:,0] 
font_label = label[:,1]

data_label = data_label.reshape(-1,1)
font_label = font_label.reshape(-1,1)
"""
length = len(data)

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 1
        self.img_shape = (self.channels,self.img_rows, self.img_cols)
        self.num_classes = num_classes
        self.font_classes = num_fonts
        self.latent_dim1 = 50
        self.latent_dim2 = 50
        #self.latent_dim = self.latent_dim1+self.latent_dim2
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label1 = Input(shape=(1,))
        label2 = Input(shape=(1,))
        img = self.generator([noise, label1, label2])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label1,label2])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label1,label2], valid)
        plot_model(self.combined, to_file='mixed.png', show_shapes=True, show_layer_names=False)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):
        
        noise = Input(shape=(self.latent_dim,))
        label1 = Input(shape=(1,),dtype='int32')
        label2 = Input(shape=(1,),dtype='int32')
        
        

        label_embedding1 = Flatten()(Embedding(num_classes, self.latent_dim1)(label1))
        label_embedding2 = Flatten()(Embedding(num_fonts, self.latent_dim2)(label2))
        
        label_embedding = concatenate([label_embedding1,label_embedding2])

        model_input = concatenate([noise, label_embedding])

        hid = Dense(128 * 8 * 8, activation='relu')(model_input)    
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
        hid = Reshape((128, 8, 8))(hid)
  
        hid = Conv2D(32, kernel_size=4, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)    
        hid = LeakyReLU(alpha=0.1)(hid)
  
        hid = Conv2DTranspose(32, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
  
        hid = Conv2D(32, kernel_size=5, strides=1,padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)    
        hid = LeakyReLU(alpha=0.1)(hid)
  
        hid = Conv2DTranspose(32, 4, strides=2, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
  
        hid = Conv2D(32, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
  
        hid = Conv2D(32, kernel_size=5, strides=1, padding='same')(hid)
        hid = BatchNormalization(momentum=0.9)(hid)
        hid = LeakyReLU(alpha=0.1)(hid)
                      
        hid = Conv2D(1, kernel_size=5, strides=1, padding="same")(hid)
        out = Activation("tanh")(hid)
        mdl = Model(inputs=[noise, label1,label2], outputs=out)
        mdl.summary()
        plot_model(mdl, to_file='generator.png', show_shapes=True, show_layer_names=False)




        
        
        
        #model_input = multiply([noise, label_embeding])
        

        return mdl

    def build_discriminator(self):



         input_layer = Input(shape=(1,32,32))
         label1 = Input(shape=(1,))
         label2 = Input(shape=(1,))
         
         
         label_embedding1 = Flatten()(Embedding(num_classes, self.latent_dim1)(label1))
         label_embedding2 = Flatten()(Embedding(num_fonts, self.latent_dim2)(label2))
        
         label_embedding = concatenate([label_embedding1,label_embedding2])
         
         

         hid = Conv2D(32, kernel_size=3, strides=1, padding='same')(input_layer)
         hid = BatchNormalization(momentum=0.9)(hid)
         hid = LeakyReLU(alpha=0.1)(hid)
  
         hid = Conv2D(32, kernel_size=4, strides=2, padding='same')(hid)
         hid = BatchNormalization(momentum=0.9)(hid)
         hid = LeakyReLU(alpha=0.1)(hid)
  
         hid = Conv2D(32, kernel_size=4, strides=2, padding='same')(hid)
         hid = BatchNormalization(momentum=0.9)(hid)
         hid = LeakyReLU(alpha=0.1)(hid)
  
         hid = Conv2D(32, kernel_size=4, strides=2, padding='same')(hid)
         hid = BatchNormalization(momentum=0.9)(hid)
         hid = LeakyReLU(alpha=0.1)(hid)
         hid = Flatten()(hid)
  
         merged_layer = concatenate([hid, label_embedding])
         hid = Dense(512, activation='relu')(merged_layer)
         #hid = Dropout(0.4)(hid)
         out = Dense(1, activation='sigmoid')(hid)
         mdl = Model(inputs=[input_layer, label1, label2], outputs=out)
         mdl.summary()
         plot_model(mdl, to_file='discriminator.png', show_shapes=True, show_layer_names=False)
        
   
        
        
        #model_input = multiply([flat_img, label_embeding])
        
        

        

         return mdl

    def train(self, epochs, batch_size=128, sample_interval=50):
        
        
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        gen_loss = []
        dis_loss = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            
            rand = np.random.choice(length, size=10000)
            data1 = data[rand]
            data_label1 = data_label[rand]
            font_label1 = font_label[rand]
            

            #data_label = data_label.reshape(-1,1)
            num_batches = int(10000/batch_size)
            for i in range(num_batches):
                imgs = data1[i*batch_size:(i+1)*batch_size]
                labels = data_label1[i*batch_size:(i+1)*batch_size]
                labels = labels.reshape(-1,1)
                font_labels = font_label1[i*batch_size:(i+1)*batch_size]
                font_labels = font_labels.reshape(-1,1)
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                gen_imgs = self.generator.predict([noise, labels, font_labels])
                d_loss_real = self.discriminator.train_on_batch([imgs, labels, font_labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels, font_labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                sampled_labels1 = np.random.randint(0, num_classes-1, batch_size).reshape(-1, 1)
                sampled_labels2 = np.random.randint(0, num_fonts-1, batch_size).reshape(-1, 1)
                g_loss = self.combined.train_on_batch([noise, sampled_labels1, sampled_labels2], valid)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                
                
                dis_loss.append(d_loss[0])
                gen_loss.append(g_loss)


            if epoch % sample_interval == 0:
                self.sample_images(epoch)
        
        plt.show()
        plt.plot(dis_loss,'r',label='discriminator loss')
        plt.plot(gen_loss,'b',label='generator loss')
        plt.legend()
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.show()
        
        model_json = self.generator.to_json()
        with open("gan.json", "w") as json_file:
           json_file.write(model_json)
         # serialize weights to HDF5
        self.generator.save_weights("gan.h5")
        print("Saved model to disk")
       
       
    def sample_images(self, epoch):
        plt.show()
        print("\ng\n")
        r, c = 2, 5
        a = 7
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        #sampled_labels1 = np.arange(0, r*c).reshape(-1, 1)
        
        
        sampled_labels1 = np.arange(0, r*c).reshape(-1, 1)
        sampled_labels2 = a*np.ones(r*c).reshape(-1,1)

        gen_imgs = self.generator.predict([noise, sampled_labels1, sampled_labels2])
        gen_imgs = gen_imgs.reshape(10,32,32,1)
        gen_imgs = np.squeeze(gen_imgs)

        # Rescale images 0 - 1
        gen_imgs1 = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0

        
        #plt.show()
        for i  in range(r):
            for j in range(c):
                plt.subplot(r,c,cnt+1)
                temp = gen_imgs1[cnt]
                #temp = np.mean(temp,axis=2)
                #temp = gen_imgs1[cnt,:,:,0]
                #temp = np.mean(temp,axis=3)
                #temp = np.squeeze(temp)
                plt.imshow(temp, cmap='gray')
                #plt.imshow(temp)
                cnt = cnt + 1

       
if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=200, batch_size=100, sample_interval=1)
    

    