import torch
import torch.nn as nn
import time
from pathlib import Path
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = Path(__file__).resolve().parent.parent
models_folder_path = root_path/"Models"


#---------main classifier model-----#

class Classifier(nn.Module):
      def __init__(self):
            super().__init__()
            self.base_model = models.efficientnet_b0(weights = models.EfficientNet_B0_Weights.DEFAULT)

            #freeze the parameters of the base model
            for param in self.base_model.parameters():
                  param.requires_grad = False

            in_features = self.base_model.classifier[1].in_features

            self.base_model.classifier = nn.Sequential(
                  nn.Dropout(p=0.2,inplace=True),
                  nn.Linear(in_features,512),
                  nn.ReLU(),
                  nn.Linear(512,10)
            )

      def forward(self, x):

            return self.base_model(x)
            

#------initilizing model----#


def init_model(model_name = None):
      model = Classifier()
      model.to(device)
      if model_name is not None:
            model_path = models_folder_path/ f"{model_name}.pth"
            if model_path.exists():
                  model.load_state_dict(
                        torch.load(
                              model_path, 
                              map_location=device, 
                              weights_only= True
                              )
                        )
                  print(f"loaded {model_name}.pth")
            else:
                  print(f"{model_name}.pth does not exist, making a new model")
      else:
            print("no name provided, making new model")
      return model


#---------functions for training and testing model---------#


def train_model(dataloader, model, learning_rate = 0.002):
      optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
      loss_fn = nn.CrossEntropyLoss()

      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      correct_prediction = 0.0
      total_loss  = 0
      model.train()
      for batch, (image,label) in enumerate(dataloader):
            image, label =  image.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred,label)
            correct_prediction += (pred.argmax(1)==label).type(torch.float).sum().item()
            #back propagation stuffs
            loss.backward()
            optimizer.step()
             
            total_loss += loss.item()

            #print loss of current btachper 100 batch
            if batch%100 == 0:
                  current_batch = (batch+1)*len(image)
                  print(f"loss: {loss.item():>7f}  [{current_batch:>5d}/{size:>5d}]")

      avg_train_loss = total_loss/len(dataloader)
      print(f"Training Loss: {avg_train_loss:.4f}")
      accuracy =(correct_prediction/len(dataloader.dataset))*100
      print(f"Training Accuracy: {accuracy:.2f}%")

      return avg_train_loss


def test_model(dataloader,model):
      loss_fn = nn.CrossEntropyLoss()
      correct_prediction = 0.0
      test_loss =0.0
      model.eval()

      with torch.no_grad():
            for image, label in dataloader:
                  image, label =  image.to(device), label.to(device)
                  pred = model(image)

                  test_loss += loss_fn(pred,label).item()
                  
                  correct_prediction += (pred.argmax(1)==label).type(torch.float).sum().item()

      avg_loss = test_loss/len(dataloader)
      accuracy =(correct_prediction/len(dataloader.dataset))*100

      print(f"\nTest Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}\n")
      return


def train_test_model(train_dataloader, test_dataloader,model,epochs=10, learning_rate =0.001):
      start_time = time.time()
      for epoch in range(epochs):
            train_model(train_dataloader, model, learning_rate)
            test_model(test_dataloader,model)
      end_time = time.time()
      train_test_time = end_time - start_time
      print("\nFinished training")
      print(f"Total Training Time for {epochs} epochs: {train_test_time:.2f} seconds")
      print(f"Average Time per epochs: {train_test_time/epochs} seconds")
      


def save_model(model, model_name=None):
      if model_name is not None:
            save_path = models_folder_path/f"{model_name}.pth"
      else:
            save_path = models_folder_path/"default model.pth"
      torch.save(model.state_dict(),save_path)
      print(f"Saved weights to {save_path}")