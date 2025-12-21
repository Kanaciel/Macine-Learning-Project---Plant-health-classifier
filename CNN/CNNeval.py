import torch
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
from .CNNdataloader import trans_img_to_tensor, tomato_class_names

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, image_tensor):
      
      with torch.no_grad():
            if image_tensor.dim() ==3: #add batch for single image
                  image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)
            prediction = model(image_tensor)

            return prediction #raw preds valuz
            

def check_accuracy_from_dataset(model,dataset, batch_size =64):
      model.eval()
      loader = DataLoader(dataset, batch_size=batch_size, shuffle= False)
      correct_guesses = 0.0
      total_guesses = 0.0

      with torch.no_grad():
            for images,labels in loader:
                  predictions = predict_image(model ,images)
                  predicted_labels = predictions.argmax(dim=1)

                  correct_guesses += (predicted_labels == labels.to(device)).sum().item()
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

