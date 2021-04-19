import numpy as np

# kNN classifier class with different distance metrics
class kNN(object):
    def __init__(self):
        pass
    
    # train: store all the data and labels into memory
    def train(self, train_data, train_labels):
        self.X_train = train_data
        self.y_train = train_labels

    # define 2 distances
    def L1(self, test_data):
        num_test = test_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.abs(test_data[i] - self.X_train[j]).sum()
        
        return dists
    
    def L2(self, test_data):
        num_test = test_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        
        #split(p-q).2 to p"2+g2-2pq, Learn from CS231
        dists = np.sqrt((test_data**2).sum(axis=1, keepdims=True) + (self.X_train**2).sum(axis=1) - 2 * test_data.dot(self.X_train.T))
        
        return dists

    def mink(self, test_data, p_value=3):
        '''
        Compute the distance between each test point in X and each training point
        in self.X_train.
        '''
        num_test = test_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.power(np.power(test_data[i]-self.X_train[j], p_value).sum(), 1./p_value)
        
        return dists

    def cos(self, test_data):
        '''
        Compute the distance between each test point in X and each training point
        in self.X_train.
        '''
        num_test = test_data.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        
        for i in xrange(num_test):
            for j in xrange(num_train):
                dists[i, j] = np.abs(test_data[i].dot(self.X_train[j].T)) \
                / (np.square(np.sum(test_data[i]**2)) * np.square(np.sum(self.X_train[j]**2)))

        return dists
    
    # test: choose the samples which have the k'th shortest distance and return their labels  
    def test(self, X, k=1, mode='L2'):
        """
        Predict labels for test data using this classifier.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
        test data, where y[i] is the predicted label for the test point X[i].  
        """
        if mode == 'L1':
            dists = self.L1(X)
        if mode == 'L2':
            dists = self.L2(X)

        predicts = self.predict(dists, k=k)
        
        return predicts

    def predict(self, dists, k=1):
        """
        # return: shape (num_test, ) the predicted label for testing data  
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])][:k]

            y_pred[i] = np.argmax(np.bincount(closest_y))

        return y_pred