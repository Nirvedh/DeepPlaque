import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import threading
import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.preprocessing.image import array_to_img, img_to_array, load_img,ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import get_dilated_unet
#from scipy.misc import imresize
import matplotlib.pyplot as plt
import SimpleITK as sitk
from resize import resizeimage, resizeimageactual
from getpath import find
from random import shuffle
from readdata import getdata, getframes
from losses import dice_coef
from keras import backend as K
from crop import getwindowdef,getwindowdefv2, getwindow,getwindowv2,getguidance
import csv
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))
WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 4

def grey2rgb(img):
  new_img = []
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      new_img.append(list(img[i][j])*3)
  new_img = np.array(new_img).reshape(img.shape[0], img.shape[1], 3)
  return new_img
class ThreadSafeIterator:

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
    A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))

    return g


@threadsafe_generator
def train_generator(df,model):
    while True:
        shuffle_indices = np.arange(len(df))
        shuffle_indices = np.random.permutation(shuffle_indices)
        shuffle(df)
        #print(df[0])        
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            
            end = min(start + BATCH_SIZE, len(df))
            referenceimg = sitk.ReadImage('../referenceImages/pat20050_L_BULB_LONG_cont_20130910085802_bmode_33.mha')

            #ids_train_batch = df.iloc[shuffle_indices[start:end]]
            for _id in df[start:end]:
                img =   sitk.ReadImage(_id)
                x = (np.random.random()-0.5)*64
                y = (np.random.random()-0.5)*64
                padding=64+(np.random.random()-0.5)*20
                #angle = (np.random.random()-0.5)*20
                #scalex = 1+(np.random.random()-0.5)/10
                #scaley = 1+(np.random.random()-0.5)/10
                #shearx = (np.random.random()-0.5)/5
                #sheary = (np.random.random()-0.5)/5
                img = resizeimage([2*WIDTH, 2*HEIGHT],0,img,referenceimg)
                #img = resizeimageactual([2*WIDTH, 2*HEIGHT],0,img,referenceimg,x,y,angle,scalex,scaley,shearx,sheary)
                #print(img.GetSize())

                #print(img.shape)
                #print('/data/data_us4/home/plaquestudy/nhm_processing/Carvana/train/{}.jpg'.format(_id))                
                maskv = sitk.ReadImage(_id.replace('bmode','Segmentation').replace('train_','train_segmentation_'))
                #print(mask)
                #print('/data/data_us4/home/plaquestudy/nhm_processing/Carvana/train_masks/{}_mask.gif'.format(_id))
                maskv = resizeimage([2*WIDTH, 2*HEIGHT],0,maskv,referenceimg)
                #maskv = resizeimageactual([WIDTH, HEIGHT],0,maskv,referenceimg,x,y,angle,scalex,scaley,shearx,sheary)
                params = getwindowdefv2(maskv,x,y,padding,[2*WIDTH, 2*HEIGHT])
                maskv2 = getwindowv2([2*WIDTH, 2*HEIGHT],1,maskv,referenceimg,params)
                maskv=resizeimage([WIDTH,HEIGHT],1,maskv2,maskv)
                
                mask=sitk.GetArrayFromImage(maskv)
                mask=mask.reshape(mask.shape[1],mask.shape[2],mask.shape[0])
                
                
                img2 = getwindowv2([2*WIDTH, 2*HEIGHT],0,img,referenceimg,params)
                img=resizeimage([WIDTH,HEIGHT],0,img2,img)
                img = sitk.GetArrayFromImage(img)
                #img = (img - img.min()) / (img.max() - img.min())
                #print(img[0,1])
                img=img.reshape(img.shape[1],img.shape[2],img.shape[0])
                #print(mask.shape)
                #print(img.shape)
                #mask = np.expand_dims(mask, axis=-1)
                #print(mask.ndim)
                #perf = model.evaluate([img],[mask],verbose=0)
                #print(_id,perf[1])

                assert mask.ndim == 3
                
                # === You can add data augmentations here. === #
                num = np.random.random()
                if num < 0.25:
                    img, mask = img[:, ::-1, :], mask[:, ::-1, :]  # random horizontal flip
                elif num<0.5:
                    img, mask = img[::-1, :, :], mask[::-1, :, :]  # random vertical flip
                elif num<0.75:
                    img, mask = img[::-1, ::-1, :], mask[::-1, ::-1, :]  # flip both
                #img = (img - img.min()) / (img.max() - img.min())
                x_batch.append(img)
                y_batch.append(mask)
            
            x_batch = np.array(x_batch, np.float32)/255.
            x=np.array(x_batch, np.float32)
            #x_batch = (x - x.min()) / (x.max() - x.min())
            y_batch = np.array(y_batch, np.float32)
            #print('x shape',x_batch.shape)
            #print('y shape',y_batch.shape)
            #perf = model.evaluate(x_batch,y_batch,verbose=0)
            #print(_id,perf)

            yield x_batch, y_batch


@threadsafe_generator
def valid_generator(df,xind=0,yind=0):
    while True:
        for start in range(0, len(df), BATCH_SIZE):
            x_batch = []
            y_batch = []
            end = min(start + BATCH_SIZE, len(df))
            #ids_train_batch = df.iloc[start:end]
            referenceimg = sitk.ReadImage('../referenceImages/pat20050_L_BULB_LONG_cont_20130910085802_bmode_33.mha')

            for _id in df[start:end]:
                img =  sitk.ReadImage(_id)
                #x = (np.random.random()-0.5)*56*2
                #y = (np.random.random()-0.5)*56*2
                #padding=65+(np.random.random()-0.5)*20
                #x=13
                #y=0
                padding=64

                maskv = sitk.ReadImage(_id.replace('bmode','Segmentation').replace('train_','train_segmentation_').replace('test_','test_segmentation_'))
                #print(mask)
                #print('/data/data_us4/home/plaquestudy/nhm_processing/Carvana/train_masks/{}_mask.gif'.format(_id))
                maskv = resizeimage([2*WIDTH, 2*HEIGHT],0,maskv,referenceimg)
                
                params = getwindowdefv2(maskv,xind,yind,padding,[2*WIDTH, 2*HEIGHT])
                maskv2 = getwindowv2([2*WIDTH, 2*HEIGHT],1,maskv,referenceimg,params)
                maskv=resizeimage([WIDTH,HEIGHT],1,maskv2,maskv)
                
                mask=sitk.GetArrayFromImage(maskv)
                mask=mask.reshape(mask.shape[1],mask.shape[2],mask.shape[0])
                
                img = resizeimage([2*WIDTH, 2*HEIGHT],0,img,referenceimg)
                img2 = getwindowv2([2*WIDTH, 2*HEIGHT],0,img,referenceimg,params)
                img=resizeimage([WIDTH,HEIGHT],0,img2,img)
                
                
                img=sitk.GetArrayFromImage(img)
                #img = (img - img.min()) / (img.max() - img.min())
                img=img.reshape(img.shape[1],img.shape[2],img.shape[0])
                #print(mask.shape)
                #mask = np.expand_dims(mask, axis=-1)

                #kv = cv2.VideoCapture('/data/data_us4/home/plaquestudy/nhm_processing/Carvana/train_masks/{}_mask.gif'.format(_id), cv2.IMREAD_GRAYSCALE)
		#val,mask=maskv.read()
                #print(mask)
                #print('/data/data_us4/home/plaquestudy/nhm_processing/Carvana/train_masks/{}_mask.gif'.format(_id))
                #mask = np.expand_dims(mask, axis=-1)
                assert mask.ndim == 3
                #img = (img - img.min()) / (img.max() - img.min())
                x_batch.append(img)
                y_batch.append(mask)
                
            x_batch = np.array(x_batch, np.float32)/255.
            x=np.array(x_batch, np.float32)
            #x_batch = (x - x.min()) / (x.max() - x.min())
            y_batch = np.array(y_batch, np.float32)

            yield x_batch, y_batch


if __name__ == '__main__':
    ids_train=[]
    ids_valid=[]
    ids_test=[]
    for path in find('*.mha','../train_parts_final/'):
        ids_train.append(path)

    for path in find('*.mha','../test_parts_final/'):
        ids_test.append(path)

    ids_train = getdata(ids_train)
    ids_train, ids_valid = train_test_split(ids_train, test_size=0.05)
    inputwriter = csv.writer(open('ids_train_v2.csv','w'))
    inputwriter2 = csv.writer(open('ids_valid_v2.csv','w'))
    ids_train = getframes('../train_parts_final/',ids_train)
    ids_valid = getframes('../train_parts_final/',ids_valid)
    for entry in ids_train:
        inputwriter.writerow([entry])
    for entry in ids_valid:
        inputwriter2.writerow([entry])
    '''inputreader = csv.reader(open('ids_train.csv','r'))
    inputreader2 = csv.reader(open('ids_valid.csv','r'))
    for line in inputreader:
        ids_train.append(line[0])
    for line in inputreader2:
        ids_valid.append(line[0])'''


    model = get_dilated_unet(
        input_shape=(256, 256, 1),
        mode='cascade',
        filters=32,
        n_class=1
    )
    
    
    class MyCallback(Callback):
        def __init__(self, test_data,steps):
            self.test_data = test_data
            self.steps=steps
        

        def on_epoch_end(self, epoch, logs={}):
            #print("here ",epoch)
            #if(epoch>0):
            perf = self.model.evaluate_generator(generator=self.test_data,steps=self.steps,verbose=0)
            print("performance ",self.model.metrics_names,perf)
            return


    class CustomStopper(EarlyStopping):
        def __init__(self, monitor='val_loss',min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 4): # add argument for starting epoch
            super(CustomStopper, self).__init__(monitor=monitor,patience=patience,min_delta=min_delta,mode=mode)
            self.start_epoch = start_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch > self.start_epoch:
                super().on_epoch_end(epoch, logs)

        
    class CustomReduce(ReduceLROnPlateau):
        def __init__(self, monitor='val_loss',factor=0, patience=0, verbose=0, epsilon=0.1,mode='auto',start_epoch = 4): # add argument for starting epoch
            super(CustomReduce, self).__init__(monitor=monitor,factor=factor,patience=patience,verbose=verbose,epsilon=epsilon,mode=mode)
            self.start_epoch = start_epoch

        def on_epoch_end(self, epoch, logs=None):
            if epoch > self.start_epoch and epoch>0:
                super().on_epoch_end(epoch, logs)


    callbacks = [CustomStopper(monitor='val_dice_coef',
                               patience=12,
                               verbose=1,
                               min_delta=1e-4,
                               mode='max'),
                 ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='semiautomatic_unet_reviewer1_iter2.hdf5',
                                 save_best_only=True,
                                 mode='max'),
                 MyCallback(valid_generator(ids_test,0,0),np.ceil(float(len(ids_test)) / float(BATCH_SIZE)))]
                   
    '''callbacks = [ReduceLROnPlateau(monitor='val_dice_coef',
                                   factor=0.2,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4,
                                   mode='max'),
                 ModelCheckpoint(monitor='val_dice_coef',
                                 filepath='semiautomatic_unet_reviewer2_iter1.hdf5',
                                 save_best_only=True,
                                 mode='max'),
                 MyCallback(valid_generator(ids_test,0,0),np.ceil(float(len(ids_test)) / float(BATCH_SIZE)))]'''
              
    #model.load_weights('semiautomatic_unet_reviewer1_iter3.hdf5')
    #model.load_weights('trained_model4.h5')
    #print(np.ceil(float(len(ids_train)) / float(BATCH_SIZE)),np.ceil(float(len(ids_valid)) / float(BATCH_SIZE)))
    
    #for testx,testy in valid_generator(ids_valid):
      #print(ids_test[count])
    #out=model.predict(valid_generator(ids_valid))
     # perf = model.evaluate(testx,testy,verbose=0)
     # break

    model.fit_generator(generator=train_generator(ids_train,model),
                        steps_per_epoch=np.ceil(float(len(ids_train)) / float(BATCH_SIZE)),
                        epochs=35,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid,0,0),
                        validation_steps=np.ceil(float(len(ids_valid)) / float(BATCH_SIZE)))
    model.save('trained_dilatedsemiunet_rivewe1_iter2.h5')
    count=0
    print('Done fitting')
    '''for y in np.arange(5,7.51,2.5):
        print('y',y,round(64/12.5*y))
        yind=int(round(64/12.5*y))
        print(yind)
        for  x in  np.arange(-7.5,4.9,2.5): 
            print('x',x,round(64/12.5*x))
            xind=int(round(64/12.5*x))
            print(xind)
            perf = model.evaluate_generator(generator=valid_generator(ids_test,xind,yind),steps=np.ceil(float(len(ids_test)) / float(BATCH_SIZE)),verbose=0)
            print("performance ",model.metrics_names,perf)
    perf = model.evaluate_generator(generator=valid_generator(ids_test,0,0),steps=np.ceil(float(len(ids_test)) / float(BATCH_SIZE)),verbose=0)
    print("performance ",model.metrics_names,perf)'''
    ''' savepath='../results_semi_unet/' 
    new_list=ids_test#ids_train+ids_valid
    for testx,testy in valid_generator(ids_test):
      #print(ids_test[count])
    #out=model.predict(valid_generator(ids_valid))
      perf = model.evaluate(testx,testy,verbose=0)
      print(new_list[count],count,perf[1])
      out=model.predict(testx)
      ind=str(count)
      seg=(out[0]> 0.5)*1.0
      #print(K.eval(dice_coef(testy,seg)))
      seg=seg.reshape(seg.shape[2],seg.shape[0],seg.shape[1])
      segImage=sitk.GetImageFromArray(seg)
      sitk.WriteImage(segImage,savepath+ind+'_seg.mha')
      
      seg=testy[0]
      seg=seg.reshape(seg.shape[2],seg.shape[0],seg.shape[1])
      segImage=sitk.GetImageFromArray(seg)
      sitk.WriteImage(segImage,savepath+ind+'_grnd.mha')

      seg=testx[0]
      seg=seg.reshape(seg.shape[2],seg.shape[0],seg.shape[1])
      segImage=sitk.GetImageFromArray(seg)
      sitk.WriteImage(segImage,savepath+ind+'.mha')
      #plt.imshow(grey2rgb(testx[0]))
      #plt.show()
      #plt.imshow(grey2rgb(testy[0]))
      #plt.show()
      #plt.imshow(grey2rgb((out[0]> 0.5)*1.0))
      #plt.show()


      count=count+1
      if(count==201):
          break'''
    #plt.imshow(grey2rgb(testx[0]))
    #plt.show()
    #plt.imshow(grey2rgb(testy[0]))
    #plt.show()
    #plt.imshow(grey2rgb((out[0]> 0.5)*1.0))
    #plt.show()
