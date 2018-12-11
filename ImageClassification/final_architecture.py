#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import math
import os
import PIL
from PIL import Image
from scipy.misc import imread
import matplotlib.pyplot as plt
import random
import cv2
import sys

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (5.0, 5.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
os.environ["CUDA_VISIBLE_DEVICES"] = ''


# In[2]:


def get_data(num_classes=250, res=128, flip=True, color_invert=False, center=True):
    # root_dir = "data/png{}/".format("" if res is None else res)
    # root_dir = "/home/sb4019/project/png{}/".format("" if res is None else res)
    root_dir = "../png/"

    num_train = 48
    # num_train = 96 if flip else 48
    num_val = 16
    num_test = 16
    # num_val = 16
    # num_test = 16

    labels = []
    
    X_train = np.zeros((num_classes * num_train , res, res, 1), dtype=np.float32)
    y_train = np.repeat(np.arange(num_classes), num_train)
    
    X_val = np.zeros((num_classes * num_val, res, res, 1), dtype=np.float32)
    y_val = np.repeat(np.arange(num_classes), num_val)
    
    X_test = np.zeros((num_classes * num_test, res, res, 1), dtype=np.float32)
    y_test = np.repeat(np.arange(num_classes), num_test)

    print (X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    
    classes = 0
    train_index = 0
    val_index = 0
    test_index = 0
    
    for node in sorted(os.listdir(root_dir)):
        if "DS_" in node:
            continue
        if os.path.isfile(root_dir + node):
            continue
        
        labels.append(node)
        label_path = root_dir + node + "/"
        
        num_images = 0
        for im_file in sorted(os.listdir(label_path)):
            if "DS_" in im_file:
                continue
            # im_data = load_image(label_path + im_file).reshape(res, res, 1)
            image = load_image(label_path + im_file)
            kernel = np.ones((5,5), np.uint8)
            image = cv2.erode(image, kernel, iterations = 1)
            im_data = cv2.resize(image, (res, res))
            # cv2.imshow("lala", im_data)
            # cv2.waitKey(5000)
            # cv2.imwrite("some.jpg", im_data)
            im_data = im_data.reshape(res, res, 1)
            
            if color_invert:
                im_data = -1 * im_data + 255
            
            if num_images < num_train:
                X_train[train_index] = im_data
                train_index += 1
                
                if flip:
                    X_train[train_index] = np.flip(im_data, axis=1)
                    train_index += 1
                    num_images += 1
                    
            elif num_images < num_train + num_val:
                X_val[val_index] = im_data
                val_index += 1
            else:
                if test_index < num_classes * num_test:
                    X_test[test_index] = im_data
                    test_index += 1
                
            num_images += 1
                
        classes += 1
        if classes == num_classes:
            break

    if center:
        X_train -= np.mean(X_train, axis=0)
        X_val -= np.mean(X_val, axis=0)
        X_test -= np.mean(X_test, axis=0)
    return X_train, y_train, X_val, y_val, X_test, y_test, labels

def load_image(path):
    im_data = imread(path, mode='L')
    return im_data


# In[3]:


X_train, y_train, X_val, y_val, X_test, y_test, labels = get_data()


# In[4]:


plt.imshow(np.squeeze(X_train[0]))
y_train[0]
labels[0]


# In[5]:


def resnet_dropout(X, y, layer_depth=4, num_classes=250, reg=1e-2, is_training=True):
    # RESnet-ish
    l2_reg = tf.contrib.layers.l2_regularizer(reg)

    """
    Input: 128x128x1
    Output: 64x64x64
    """
    d0 = tf.layers.dropout(X, rate=0.5, training=is_training)
    c0 = tf.layers.conv2d(d0, 64, [7, 7], strides=[2, 2], padding='SAME', kernel_regularizer=l2_reg)
    c0 = tf.layers.batch_normalization(c0, training=is_training)
    match_dimensions = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 64, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        r = c0 + b2
        c0 = tf.nn.relu(r)
    
    """
    Input: 64x64x64
    Output: 32x32x128
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 128, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 128, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 128, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 32x32x128
    Output: 16x16x256
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 256, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 256, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 256, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)

    """
    Input: 16x16x256
    Output: 8x8x512
    """
    downsample = True
    for i in range(layer_depth):
        c1 = tf.layers.conv2d(c0, 512, [3, 3], 
                              strides=([2, 2] if downsample else [1, 1]),
                              padding='SAME',
                              kernel_regularizer=l2_reg)
        b1 = tf.layers.batch_normalization(c1, training=is_training) #bn
        h1 = tf.nn.relu(b1) #relu
        c2 = tf.layers.conv2d(h1, 512, [3, 3], padding='SAME', kernel_regularizer=l2_reg) #conv
        b2 = tf.layers.batch_normalization(c2, training=is_training) #bn
        if downsample:
            c0_proj = tf.layers.conv2d(c0, 512, [1, 1], padding='SAME', kernel_regularizer=l2_reg)
            c0_proj = tf.layers.average_pooling2d(c0_proj, (2, 2), (2, 2))
            r = c0_proj + b2
            downsample = False
        else:
            r = c0 + b2
        c0 = tf.nn.relu(r)
    
    p1 = tf.layers.average_pooling2d(c0, (8, 8), (1,1))
    p1_flat = tf.reshape(p1, [-1, 512])
    d1 = tf.layers.dropout(p1_flat, rate=0.2, training=is_training)
    y_out = tf.layers.dense(d1, num_classes, kernel_regularizer=l2_reg)
    
    return y_out


# In[6]:


res = 128
num_classes=250
tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, res, res, 1])
y = tf.placeholder(tf.int64, [None])
is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)
reg = tf.placeholder(tf.float32)

# y_out = naive_model(X, y)
y_out = resnet_dropout(X, y, layer_depth=2, num_classes=num_classes, is_training=is_training, reg=reg)
# print (y_out.shape)
mean_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(y, num_classes), logits=y_out))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

# batch normalization in tensorflow requires this extra dependency
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)
reg_val = 1e-2
learning_rate = 1e-3


# In[7]:


def run_model(session, predict, loss_val, Xd, yd,
              epochs=1, batch_size=256, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    correct_prediction = tf.equal(tf.argmax(predict,1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)

    training_now = training is not None
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [mean_loss,correct_prediction,accuracy]
    if training_now:
        variables[-1] = training
    
    # counter 
    iter_cnt = 0
    for e in range(epochs):
                # keep track of losses and accuracy
        correct = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                        # generate indicies for the batch
            start_idx = (i*batch_size)%X_train.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:], y: yd[idx], is_training:training_now,
                         lr : learning_rate, reg: reg_val}
                        
                        
            # get batch size
            actual_batch_size = yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            correct += np.sum(corr)
            
            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}".format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
            iter_cnt += 1
        total_correct = correct/Xd.shape[0]
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}".format(total_loss,total_correct,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss,total_correct


# In[8]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_losses = []
train_acc = []
val_losses = []
val_acc = []
max_acc = 0
saver = tf.train.Saver()


# In[ ]:


epochs = 20
reg_val = 1e-1
learning_rate = 3e-3

# tf.reset_default_graph()  
# imported_meta = tf.train.import_meta_graph("../../../0.5429333333333334model.ckpt.meta")  

# saver = tf.train.import_meta_graph("../../../0.5429333333333334model.ckpt.meta")  
# # sess = tf.Session()
# saver.restore(sess,'../../../0.5429333333333334model.ckpt')
# anss = y_val.copy()
# for i in range(0, X_val.shape[0]):
    
#     correct_prediction = tf.equal(tf.argmax(y_out,1), y)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     variables = [mean_loss, correct_prediction, y ]

#     loss, corr, _ans = sess.run(variables, feed_dict={X:X_val[i].reshape(1, 128, 128, 1), y: y_val[i].reshape(1), is_training:False})
#     # cv2.imshow(labels[y_val[i]] + " as " + labels[_ans[0]], X_val[i])
#     # cv2.waitKey(2000)
#     print("Expected ", labels[anss[i]], "Predicted ", labels[_ans[0]], corr)
    # tf.print( y_out, output_stream = a )
    # print (a)



for i in range(epochs):
    loss, acc = run_model(session=sess,
                                 predict=y_out,
                                 loss_val=mean_loss,
                                 Xd=X_train,
                                 yd=y_train,
                                 epochs=1,
                                 batch_size=256,
                                 print_every=5,
                                 training=train_step,
                                 plot_losses=False)
    train_losses.append(loss)
    train_acc.append(acc)
    # loss, acc = run_model(session=sess,
    #                              predict=y_out,
    #                              loss_val=mean_loss,
    #                              Xd=X_train,
    #                              yd=y_train,
    #                              epochs=1,
    #                              batch_size=256,
    #                              print_every=5,
    #                              training=None,
    #                              plot_losses=False)
    # print(tf.Print(y_out, [y_out]))
    # print(y_train)
    loss, acc = run_model(session=sess,
                             predict=y_out,
                             loss_val=mean_loss,
                             Xd=X_val,
                             yd=y_val,
                             epochs=1,
                             batch_size=256,
                             print_every=5,
                             training=None,
                             plot_losses=False)
    if not val_acc:
        save_path = saver.save(sess, "../"+str(acc)+"model.ckpt")
    elif acc > max_acc:
        save_path = saver.save(sess, "../"+str(acc)+"model.ckpt")
    max_acc = max(max_acc, acc)
    val_losses.append(loss)
    val_acc.append(acc)


# # # In[ ]:




