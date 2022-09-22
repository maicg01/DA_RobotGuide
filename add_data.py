from types import new_class
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

directory = 'add_data'
classes = ['false', 'true']

# print(os.path.abspath(os.getcwd())) #duong dan hien tai

for i in classes:
    path = os.path.join(directory, i) # path = F:\python\CODE_thu\directory\i
    for img in os.listdir(path): #liet ke anh trong duong dan path
        img_array = cv2.imread(os.path.join(path,img))
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break

# height, weight = np.shape(img_array)
# print(height, weight)

start_row,start_col= 1085,446
end_row,end_col= 1203,590
cropped=img_array[start_col:end_col,start_row:end_row]
# h, w = np.shape(cropped)
# print(h,w)
# plt.imshow(cropped, cmap='gray')
# plt.show()

training_data = []
def create_training_data():
    for i in classes:
        path = os.path.join(directory,i)
        class_num = classes.index(i)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_img = img_array[start_col:end_col,start_row:end_row]
                training_data.append([new_img, class_num])
                # plt.imshow(new_img, cmap='gray')
                # plt.show()
            except Exception as err:
                pass

create_training_data()
# print(training_data)
import random
random.shuffle(training_data) #tron anh len
# print(training_data)

#tao tap anh va nhan
X=[] #tap anh
y=[] #tap nhan

for img, label in training_data:
    X.append(img)
    y.append(label)

# print(np.shape(X[-1]))
# print(type(X))
X = np.array(X)
y = np.array(y)
# print(type(X))
# print(type(y))
print(X.shape)
print(y.shape)

import pickle
#tao file de luu
pickle_out = open('X_CNN_add.pickel', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y_CNN_add.pickel', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()