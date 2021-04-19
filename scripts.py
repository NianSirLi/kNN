import random
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import struct

from dataloader import load_mnist_data, load_cifar10_data
from model import kNN

#from features import *

# 当前文件的绝对路径
this_dir = os.path.dirname(__file__)
classifier = kNN()

# 只用数据集中 5000 个训练样本和 500 个测试样本
NUMBER_TRAINING = 5000
NUM_TEST = 500

ALL_KS = [1, 3, 5, 7] # 所有选择的 k 值
ALL_DIS = ['L1', 'L2'] # 所有选择的 distance metrics

NUM_FOLDS = 2 # k 折交叉认证

def using_mnist():
    mnist_dir = os.path.join(this_dir, 'data', 'mnist')

    # train_labels.shape : 60000 x
    # test_labels: 10000 x
    train_data, train_labels, test_data, test_labels = load_mnist_data(mnist_dir)
    train_data = np.reshape(train_data, (train_data.shape[0], -1)) # 60000 x 28 x 28 -> 60000 x 784
    test_data = np.reshape(test_data, (test_data.shape[0], -1)) # 10000 x 28 x 28 -> 10000 x 784
    train_data = train_data[range(NUMBER_TRAINING)] # training set downsample: 60000 -> 5000
    train_labels = train_labels[range(NUMBER_TRAINING)]
    test_data = test_data[range(NUM_TEST)] # test set downsample: 10000 -> 500
    test_labels = test_labels[range(NUM_TEST)]

    # 将训练集和测试集划分为 k 部分，进行 k 折交叉认证
    train_data_folds = np.array(np.array_split(train_data, NUM_FOLDS))
    train_labels_folds = np.array(np.array_split(train_labels, NUM_FOLDS))

    results = {}
    for k in ALL_KS:
        for dis in ALL_DIS:
            setting = 'k = {:d} dis = {:s}'.format(k, dis)
            for n in range(NUM_FOLDS): # 每次的 n 就是验证集
                
                accs = []
                combinat = [x for x in range(NUM_FOLDS) if x != n] # 得到一个含 k-1 个折的迭代器
                cross_train_data = np.concatenate(train_data_folds[combinat]) # 将 k-1 个折拼在一起
                cross_train_labels = np.concatenate(train_labels_folds[combinat])
                cross_val_data = train_data_folds[n]
                cross_val_labels = train_labels_folds[n]
                
                classifier.train(cross_train_data, cross_train_labels)
                
                print('k folds looping...')
                # test 过程是很慢的，采用了 k 折交叉验证
                cross_val_pred = classifier.test(cross_val_data, k=k, mode=dis)

                num_hits = np.sum(cross_val_pred == cross_val_labels)
                acc = 1. * num_hits / cross_val_labels.shape[0]
                accs.append(acc)
            
            # 对 k 次实验（k-folds）的结果取平均
            print('Setting: {:s}, Accuracy: {:f}\n'.format(setting, np.array(accs).mean()))
            mean_acc = np.array(accs).mean()
            results[setting] = mean_acc
            
    for key, val in sorted(results.iteritems(), key=lambda kv:(kv[1],kv[0]), reverse=True):
        print('Setting:{:s}, Accuracy:{:f}'.format(key, val))


def using_cifar10():
    cifar_10_dir = os.path.join(this_dir, '..', 'data', 'cifar_10')
    
    # load training and testing data
    train_data, train_labels, test_data, test_labels = load_cifar10_data(cifar_10_dir)

    # Two alternatives here:
    # 1. Simply flatten raw image data
    # 2. Use feature extractor

    # Simply reshape raw image data
    #train_data = np.reshape(train_data, (train_data.shape[0], -1))
    #test_data = np.reshape(test_data, (test_data.shape[0], -1))

    # Use feature extractor
    num_color_bins = 10 # Number of bins in the color histogram
    feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
    train_data = extract_features(train_data, feature_fns)
    test_data = extract_features(test_data, feature_fns)

    train_data = train_data[range(NUMBER_TRAINING)]
    train_labels = train_labels[range(NUMBER_TRAINING)]

    test_data = test_data[range(NUM_TEST)]
    test_labels = test_labels[range(NUM_TEST)]

    classifier.train(train_data, train_labels)

    # do testing
    test_pred = classifier.test(test_data, k=1, mode='l2')
    num_hits = np.sum(test_labels==test_pred)
    acc = 1. * num_hits / NUM_TEST

    print('When use k=1, L2 distance setting')
    print('Accuracy is {:f}\n'.format(acc))

    print('We perform cross validation to search for the best hyper-param combination\n')

    train_data_folds = np.array(np.array_split(train_data, NUM_FOLDS))
    train_labels_folds = np.array(np.array_split(train_labels, NUM_FOLDS))

    results = {}
    for k in ALL_KS:
        for dis in ALL_DIS:
            setting = 'k={:d} dis={:s}'.format(k,dis)
            for n in range(NUM_FOLDS):
                
                accs = []
                combinat = [x for x in range(NUM_FOLDS) if x != n]
                cross_train_data = np.concatenate(train_data_folds[combinat])
                cross_train_labels = np.concatenate(train_labels_folds[combinat])
                cross_val_data = train_data_folds[n]
                cross_val_labels = train_labels_folds[n]

                classifier.train(cross_train_data, cross_train_labels)
                cross_val_pred = classifier.test(cross_val_data, k=k, mode=dis)
                num_hits = np.sum(cross_val_pred==cross_val_labels)
                acc = 1. * num_hits / cross_val_labels.shape[0]
                accs.append(acc)
                
            print('Setting: {:s}, Accuracy: {:f}'.format(setting, np.array(accs).mean()))

            mean_acc = np.array(accs).mean()
            results[setting] = mean_acc
            
    for key, val in sorted(results.iteritems(), key=lambda kv:(kv[1],kv[0]), reverse=True):
        print('Setting:{:s}, Accuracy:{:f}'.format(key, val))

using_mnist()
