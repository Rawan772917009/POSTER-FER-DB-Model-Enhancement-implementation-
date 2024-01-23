import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import torch
import os
import argparse
import torch.nn.functional as F
from PIL import Image
from data_preprocessing.dataset_raf import RafDataSet
from data_preprocessing.dataset_affectnet import Affectdataset
from data_preprocessing.dataset_affectnet_8class import Affectdataset_8class
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix

from utils import *
from models.emotion_hyp import pyramid_trans_expr
from sklearn.metrics import confusion_matrix
from data_preprocessing.plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt
from skimage import io
import os 
device = 'cuda' if torch.cuda.is_available() else "cpu"
model = pyramid_trans_expr(img_size=224, num_classes=7)
print("Loading pretrained weights...", "checkpoint/rafdb_best.pth")
checkpoint = torch.load('checkpoint/rafdb_best.pth', map_location=torch.device('cpu'))
checkpoint = checkpoint["model_state_dict"]
model = load_pretrained_weights(model, checkpoint)
model = model.to(device)
testfolder ="testfolder"
test_images_path = os.listdir(testfolder)
test_images =[]
for image_path in test_images_path:
    image = io.imread(os.path.join(testfolder,image_path))
    # Define a transformation to convert the image to a PyTorch tensor
    data_transforms_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # Apply the transformation to the image
    tensor_image = data_transforms_test(image)
    test_image = tensor_image.unsqueeze(0)
    test_images.append(test_image)
    results = []
with torch.no_grad():
    model.eval()
    for img in test_images:
        output, features = model(img.to(device))
        _, predict = torch.max(output, 1)
        predicted_label = predict.item()
        results.append(predicted_label)
        emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
        true_label_path= "data/raf-basic/EmoLabel/list_patition_label.txt"
with open(true_label_path ,'r') as file:
    file = file.readlines()
file = [x[:-1].split(" ") for x in file if x.find("test")==0]
test_path = []
for path in test_images_path:
    test_path.append(path.replace("_aligned",""))
    true_labels=[]
predicted_labels = []
for result , path in zip(results,test_path):
        for f_path in file:
            if path == f_path[0]:
                true_labels.append(int(f_path[1])-1)
                predicted_labels.append(result)
                #print(path,result,f_path[0],int(f_path[1])-1)
                predicted_labels = torch.tensor(predicted_labels)
true_labels = torch.tensor(true_labels)
correct_or_not = torch.eq(predicted_labels, true_labels)
acc= correct_or_not.sum()/len(correct_or_not)
acc = np.around(acc.numpy(), 4)
cm = confusion_matrix(true_labels, predicted_labels)
cm = np.array(cm)
labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]  #
plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)