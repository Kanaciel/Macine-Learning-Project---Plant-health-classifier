import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from .CNNdataloader import trans_img_to_tensor, tomato_class_names
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, image_tensor):
      
      with torch.no_grad():
            if image_tensor.dim() ==3: #add batch for single image
                  image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)
            prediction = model(image_tensor)

            return prediction #raw preds valuz
            
#get accuracy as percentage
def check_accuracy_from_dataset(model,dataset, batch_size =64):
      model.eval()
      loader = DataLoader(dataset, batch_size=batch_size, shuffle= False)
      correct_guesses = 0.0
      total_guesses = 0.0

      with torch.no_grad():
            for images,labels in loader:
                  predictions = predict_image(model ,images)
                  predicted_labels = predictions.argmax(dim=1)

                  correct_guesses += (predicted_labels == labels.to(device)).type(torch.float).sum().item()
                  total_guesses += labels.size(0)

      accuracy =  (correct_guesses/total_guesses)*100
      return accuracy



def predict_image_file(model, image_path):
      model.eval()
      image_path = Path(image_path)
      image = Image.open(image_path).convert('RGB')
      image_tensor = trans_img_to_tensor(image)
      image_tensor = image_tensor.unsqueeze(0)

      predictions = predict_image(model, image_tensor)
      predicted_label_index =predictions.argmax(dim=1).item()
      predicted_label = tomato_class_names[predicted_label_index]
     
      probs = torch.nn.functional.softmax(predictions, dim= 1)

      confidence = probs[0,predicted_label_index].item()*100

      return predicted_label, confidence

def get_confusion_matrix(model, dataset, batch_size=64):

      model.eval()
      loader = DataLoader(dataset, batch_size=batch_size, shuffle= False)

      num_classes = len(tomato_class_names)

      confusion_matrix = torch.zeros(num_classes,num_classes,dtype=torch.int64)

      with torch.no_grad():
            for images, labels in loader:
                  images, labels = images.to(device), labels.to(device)

                  predictions = predict_image(model,images)
                  predicted_labels = predictions.argmax(1)


                  #flatten the labels to 1d 
                  true_labels = labels.view(-1)
                  pred_labels = predicted_labels.view(-1)

                  for i in range(len(true_labels)):
                        true_class = true_labels[i]
                        pred_class = pred_labels[i]

                        confusion_matrix[true_class,pred_class] +=1
      
      return confusion_matrix


def get_F1_score(model, dataset,confusion_matrix = None,batch_size=64):
      if confusion_matrix == None:
            confusion_matrix = get_confusion_matrix(model, dataset,batch_size)

      num_classes = len(tomato_class_names)

      total_samples = confusion_matrix.sum().item()
      precisions= {}
      recalls = {}
      #per_class_accs = {}
      F1_scores = {}

      for i in range(num_classes):

            TP = confusion_matrix[i,i].item()
            FP = 0
            FN = 0
            for j in range(num_classes):
                  if i != j:
                        FP += confusion_matrix[j,i].item()
                        FN +=confusion_matrix[i,j].item()

            
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)

            precisions[tomato_class_names[i]] = precision
            recalls[tomato_class_names[i]] = recall


            F1= (2*precision*recall)/(precision+recall)
            F1_scores[tomato_class_names[i]] =F1

      return F1_scores,precisions,recalls


def display_confusion_matrix(confusion_matrix):

      tomato_class_names_abbreviated = [
      "Bacterial spot",
      "Early blight",
      "Late blight",
      "Leaf Mold",
      "Septoria",
      "Spider mites ",
      "Target Spot",
      "Yellow Curl",
      "mosaic virus",
      "healthy"
      ]

      fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

      sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=tomato_class_names_abbreviated, 
            yticklabels=tomato_class_names_abbreviated)
        
      ax.set_title("Confusion Matrix", fontsize=16)
      ax.set_xlabel("Predicted Labels", fontsize=10)
      ax.set_ylabel("True Labels", fontsize=10)

      plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
      plt.setp(ax.get_yticklabels(), fontsize=8)
      plt.tight_layout()

      plt.show()

