import os
import struct
import numpy as np

# 不用自己读了，以后读取 mnist 数据集就这么读就行
# 输入：data_dir:~\data\mnist(到二进制数据的上一级)
# 输出：train_data, train_labels, test_data, test_labels
def load_mnist_data(data_dir):
    print('start mnist data loading...\n')
    train_imgs_dir = os.path.join(data_dir, 'train-images.idx3-ubyte')
    train_labels_dir = os.path.join(data_dir, 'train-labels.idx1-ubyte')
    test_imgs_dir = os.path.join(data_dir, 't10k-images.idx3-ubyte')
    test_labels_dir = os.path.join(data_dir, 't10k-labels.idx1-ubyte')

    # 要先读labels，再读data
    with open(train_labels_dir, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.uint8)

        print("训练集的图片数量：" + str(train_labels.shape[0]))
    
    with open(train_imgs_dir, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        
        # train_labels.shape[0]是训练集的图片数量，图片分辨率：rows x cols
        train_data = np.fromfile(f, dtype=np.uint8).reshape(train_labels.shape[0], rows, cols).astype("float")

    with open(test_labels_dir, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.uint8)
        
        print("测试集的图片数量：" + str(test_labels.shape[0])+'\n') 

    with open(test_imgs_dir, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        test_data = np.fromfile(f, dtype=np.uint8).reshape(test_labels.shape[0], rows, cols).astype("float")

    print('finish mnist data loading!\n')
    
    # train_labels.shape : 60000 x 
    # train_data: 60000 x 28 x 28
    # test_labels: 10000 x
    # test_data: 10000 x 28 x 28
    return train_data, train_labels, test_data, test_labels


def load_cifar10_data(data_dir):
    print('start cifar10 data loading...')
    
    train_data, train_labels = [], []
    for i in xrange(5):
        file_name = 'data_batch_{:d}'.format(i+1)
        file_dir = os.path.join(data_dir, file_name)
        with open(file_dir, 'rb') as f:
            data_dict = pickle.load(f)
        data = data_dict['data']
        labels = data_dict['labels']
        # Transform into numpy data type, shape[10000,32,32,3]
        data = data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        labels = np.array(labels)

        train_data.append(data)
        train_labels.append(labels)
    
    # data shape [50000,32,32,3], label shape [50000,]
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    test_file_dir = os.path.join(data_dir, 'test_batch')
    with open(test_file_dir, 'rb') as f:
        data_dict = pickle.load(f)
    test_data = data_dict['data']
    test_labels = data_dict['labels']
    test_data = test_data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    test_labels = np.array(test_labels)

    print('finish cifar10 data loading')

    return train_data, train_labels, test_data, test_labels