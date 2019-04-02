from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np

if __name__ == '__main__':

    instances = 335
    features = 2
    test_set_proportion = 0.3
    validation_set_proportion = 0.2
    X = np.empty((instances, features))
    label_0 = np.zeros(235)
    label_1 = np.ones(100)
    y = np.concatenate([label_0, label_1])
    
    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_set_proportion)
    train_full_index, test_index = train_test_split.split(X, y).__next__()
    X_train_full, X_test = X[train_full_index], X[test_index]
    y_train_full, y_test = y[train_full_index], y[test_index]
    train_validation_split = StratifiedShuffleSplit(n_splits=1, test_size=validation_set_proportion)
    train_index, validation_index = train_validation_split.split(X_train_full, y_train_full).__next__()
    X_train, X_validation = X_train_full[train_index], X_train_full[validation_index]
    y_train, y_validation = y_train_full[train_index], y_train_full[validation_index]
    
    print('X_train_full.shape =', X_train_full.shape)
    print('y_train_full.shape =', y_train_full.shape)
    print('X_test.shape =', X_test.shape)
    print('y_test.shape =', y_test.shape)
    
    print('X_train.shape =', X_train.shape)
    print('y_train.shape =', y_train.shape)
    print('X_validation.shape =', X_validation.shape)
    print('y_validation.shape =', y_validation.shape)
