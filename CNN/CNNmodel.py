import torch
import torch.nn as nn
import time
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = Path(__file__).resolve().parent.parent
models_folder_path = root_path/"Models"


#---------functions and stuffs-----#
#abberation of god

class Classifier(nn.Module):
      def __init__(self):
            super().__init__()
            #none distinct stuffs
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2,2)
            self.dropout = nn.Dropout1d(0.1)
            self.dropout2d = nn.Dropout2d(0.1)
            #distinct stuffs
            self.conv1a = nn.Conv2d(3,32,3,1,1)
            self.conv1b = nn.Conv2d(32,32,3,1,1)
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2a = nn.Conv2d(32,64,3,1,1)
            self.conv2b = nn.Conv2d(64,64,3,1,1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv3a = nn.Conv2d(64,128,3,1,1)
            self.conv3b = nn.Conv2d(128,128,3,1,1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.conv4a = nn.Conv2d(128,256,3,1,1)
            self.conv4b = nn.Conv2d(256,256,3,1,1)
            self.bn4 = nn.BatchNorm2d(256)

            self.conv5a = nn.Conv2d(256,512,3,1,1)
            self.conv5b = nn.Conv2d(512,512,3,1,1)
            self.bn5 = nn.BatchNorm2d(512)


            self.fc1 = nn.Linear(512*8*8,512)
            self.fc2 = nn.Linear(512,256)
            self.fc3 = nn.Linear(256,10)


      def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1b(self.conv1a(x)))))
            x = self.maxpool(self.relu(self.bn2(self.conv2b(self.conv2a(x)))))
            x = self.maxpool(self.relu(self.bn3(self.conv3b(self.conv3a(x)))))
            x = self.maxpool(self.relu(self.bn4(self.conv4b(self.conv4a(x)))))
            x = self.maxpool(self.relu(self.bn5(self.conv5b(self.conv5a(x)))))
            
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            

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


def train_model(dataloader, model, learning_rate = 0.002):
      optimizer = torch.optim.Adam(model.parameters(), lr= learning_rate)
      loss_fn = nn.CrossEntropyLoss()

      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      correct_prediction = 0.0
      total_loss  = 0

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


def train_test_model(train_dataloader, test_dataloader,model,epochs, learning_rate =0.002):
      start_time = time.time()
      for epoch in range(epochs):
            train_model(train_dataloader, model, learning_rate)
      end_time = time.time()
      train_test_time = end_time - start_time
      print("\nFinished training")
      print(f"Total Training Time for {epochs} epochs: {train_test_time:.2f} seconds")
      print(f"Average Time per epochs: {train_test_time/epochs} seconds")
      test_model(test_dataloader,model)


def save_model(model, model_name=None):
      if model_name is not None:
            save_path = models_folder_path/f"{model_name}.pth"
      else:
            save_path = models_folder_path/"default model.pth"
      torch.save(model.state_dict(),save_path)
      print(f"Saved weights to {save_path}")