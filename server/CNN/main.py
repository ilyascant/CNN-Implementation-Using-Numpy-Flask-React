import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm

from Settings import *
from CNN.CNN import *

if __name__ == "__main__":
        
    DATA = np.asarray(pd.read_csv("train.csv"), dtype="float64")

    np.random.seed(0)
    np.random.shuffle(DATA)
    
    m, n = DATA.shape

    ###### TRAIN ######
    data_train = DATA[1000: m].copy() 

    y_train = data_train[:, 0].reshape(m - 1000, 1)
    X_train = data_train[:, 1:n] / 255.0
    X_train, y_train = Settings.randomize_images(X_train, y_datas=y_train)

    mean_test = X_train.mean().astype(np.float64)
    std_test = X_train.std().astype(np.float64)
    X_train = (X_train - mean_test) / (std_test)

    ###### TEST ######
    data_test = DATA[0: 1000].copy()

    y_test = data_test[:, 0].reshape(1000,1)
    X_test = data_test[:, 1:n] / 255.0
    X_test , y_test = Settings.randomize_images(X_test, y_datas=y_test)

    mean_test = X_test.mean().astype(np.float64)
    std_test = X_test.std().astype(np.float64)
    X_test = (X_test - mean_test) / (std_test)

    test_data = np.hstack((X_test, y_test))
    X_test = test_data[:,0:-1]
    X_test = X_test.reshape(len(test_data), 1, 28, 28)
    y_test = test_data[:,-1]


    ###### CNN ######
    cnn = CNN(X_train, y_train, X_test, y_test)

    model_filename = 'cnn_model_L0-01594374_R50421853.pkl'
    with open(model_filename, 'rb') as file:
        cnn = pickle.load(file)
        

    # costs = cnn.train()

    # print("Saving Learnt Values")
    # with open(f'cnn_model_L{costs[-1]:.8f}_R{int(random.random() * 1e8)}'.replace('.',"-")+'.pkl', 'wb') as file:
    #     pickle.dump(cnn, file)
    # print("Saving Completed")

        
    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]


    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X_test)), leave=True)
    for i in t:
        x = X_test[i]
        pred, prob, all_probs = cnn.predict(x, 1, 2, 2)
        digit_count[int(y_test[i])]+=1
        if pred==y_test[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))