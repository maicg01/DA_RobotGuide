import cv2
import matplotlib.pyplot as plt
import pickle
import os

# with open('X.pkl', 'rb') as f:
#     imageOriginSet = pickle.load(f)

# with open('y.pkl', 'rb') as f:
#     labelOriginSet = pickle.load(f)

train_dir =  'dataDetect'

def train(train_dir):
    X=[]
    y=[]
    #lap tung anh trong co so du lieu
    for image in os.listdir(train_dir):
        pathName = os.path.join(train_dir,image)

        img2 = cv2.imread(pathName)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        index_label = image.find(".")
        directory = image[:index_label]
        print(directory)

        X.append(img2)
        y.append(directory)
        # count = count + 1 
    print(X)
    print(y)  
    return X, y

def draw_sample_label(X,y,ypred=None):
    X = X[:12]
    y = y[:12]
    plt.subplots(3,4)
    for i in range(len(X)):
        plt.subplot(3,4,i+1)
        plt.imshow(X[i])
        if ypred is None:
            plt.title(f'y={y[i]}')
        else:
            plt.title(f'y={y[i]} ypred={ypred[i]}')
    plt.show()

X, y = train(train_dir)

draw_sample_label(X, y)