import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report
import face_recognition

import time
from datetime import datetime
import pickle
import faiss
import cv2
import matplotlib.pyplot as plt

from class_start.face_vecto import FaceVecto


face_recog = FaceVecto(model_file='./onnx/adaface_50w_batch.onnx')


# Define the path to LFW dataset
LFW_DIR = '/home/maicg/Downloads/detect_output_lfw'

# Define the path to pretrained arcFace model
MODEL_PATH = 'path/to/pretrained/model'

# Define the size of input image for the model
IMG_SIZE = (112, 112)

# Define the threshold for face verification
THRESHOLD = 0.4

# Define the transform for input image
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# # Load the pretrained model
# model = torch.load(MODEL_PATH)
# model.eval()


link_evaluation = '/home/maicg/Documents/Me/oop_faceRecog/evaluation_lfw'
# Load the pairs file in LFW dataset
pairs_file = open(os.path.join(link_evaluation, 'pairs.txt'), 'r')
pairs_lines = pairs_file.readlines()
pairs_file.close()

# # Load the names of all the people in LFW dataset
# names_file = open(os.path.join(link_evaluation, 'people.txt'), 'r')
# names_lines = names_file.readlines()
# names_file.close()
# for line in names_lines:
#     print(line)
#     names = [line.split('\t')[1].strip() for line in names_lines]

# Define the function to load image from file path
def load_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    return img

# # Define the function to calculate the cosine similarity between two feature vectors
# def cosine_similarity(x1, x2):
#     return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(emb1, emb2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(emb1, emb2)
    # print("goc ti le giua anh 1 va 2: ", output)
    return output.item()

# Initialize the arrays to store the predicted labels and ground truth labels
pred_labels = []
true_labels = []
pred_results = []
# Loop through all the pairs in the pairs file
for line in pairs_lines:
    # print(line)
    line = line.strip().split('\t')
    print(line)
    try:
        if len(line) == 3:
            name, id1, id2 = line
            id1 = int(id1)
            id2 = int(id2)
            file_path1 = os.path.join(LFW_DIR, name, f'{name}_{id1:04d}.jpg')
            file_path2 = os.path.join(LFW_DIR, name, f'{name}_{id2:04d}.jpg')
            print("file_path1: ", file_path1)
            print("file_path2: ", file_path2)
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            with torch.no_grad():
                feat1 = face_recog.face_embedding(img1)
                feat2 = face_recog.face_embedding(img2)
            similarity = cosine_similarity(feat1, feat2)
            print("type similarity: ", (similarity))
            if similarity >= THRESHOLD:
                pred_labels.append(1)
                true_labels.append(1)
                pred_results.append(similarity)
            else:
                pred_labels.append(0)
                true_labels.append(1)
                pred_results.append(similarity)
        elif len(line) == 4:
            name1, id1, name2, id2 = line
            id1 = int(id1)
            id2 = int(id2)
            file_path1 = os.path.join(LFW_DIR, name1, f'{name1}_{id1:04d}.jpg')
            file_path2 = os.path.join(LFW_DIR, name2, f'{name2}_{id2:04d}.jpg')
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            with torch.no_grad():
                feat1 = face_recog.face_embedding(img1)
                feat2 = face_recog.face_embedding(img2)
            similarity = cosine_similarity(feat1, feat2)
            if similarity >= THRESHOLD:
                pred_labels.append(1)
                true_labels.append(0)
                pred_results.append(similarity)
            else:
                pred_labels.append(0)
                true_labels.append(0)
                pred_results.append(similarity)
        else:
            name1, id1, name2, id2, name3, id3 = line
            id1 = int(id1)
            id2 = int(id2)
            file_path1 = os.path.join(LFW_DIR, name1, f'{name1}_{id1:04d}.jpg')
            file_path2 = os.path.join(LFW_DIR, name2, f'{name2}_{id2:04d}.jpg')
            file_path3 = os.path.join(LFW_DIR, name3, f'{name3}_{id3:04d}.jpg')
            img1 = cv2.imread(file_path1)
            img2 = cv2.imread(file_path2)
            img3 = cv2.imread(file_path3)
            with torch.no_grad():
                # feat1 = model(img1).cpu().numpy()[0]
                # feat2 = model(img2).cpu().numpy()[0]
                # feat3 = model(img3).cpu().numpy()[0]
                feat1 = face_recog.face_embedding(img1)
                feat2 = face_recog.face_embedding(img2)
                feat3 = face_recog.face_embedding(img3)
            similarity1 = cosine_similarity(feat1, feat2)
            similarity2 = cosine_similarity(feat1,feat3)
            if similarity1 >= THRESHOLD:
                pred_labels.append(1)
                true_labels.append(0)
                pred_results.append(similarity)
            else:
                pred_labels.append(0)
                true_labels.append(0)
                pred_results.append(similarity)
            
            if similarity2 >= THRESHOLD:
                pred_labels.append(1)
                true_labels.append(0)
                pred_results.append(similarity)
            else:
                pred_labels.append(0)
                true_labels.append(0)
                pred_results.append(similarity)
    except:
        print("==through==")
        continue

# Calculate the accuracy and AUC score
accuracy = accuracy_score(true_labels, pred_labels)
fpr, tpr, thresholds = roc_curve(true_labels, pred_results)
auc_score = auc(fpr, tpr)

# Vẽ ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % auc_score)
plt.plot([0, 1], [0, 1], 'k--', label='Random guessing')
plt.xlim([-0.5, 1.0])
plt.ylim([-0.5, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()



#Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'AUC score: {auc_score:.4f}')

with open('X.pkl', 'wb') as f:
    pickle.dump(true_labels, f)  

with open('y.pkl', 'wb') as f:
    pickle.dump(pred_results, f) 