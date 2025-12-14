import torch
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#abberation of god
class Classifier(nn.Module):
      def __init__(self):
            super().__init__()
            #none distinct stuffs
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2,2)
            self.dropout = nn.Dropout1d(0.05)
            #distinct stuffs
            self.conv1 = nn.Conv2d(3,32,3,1,1)
            self.bn1 = nn.BatchNorm2d(32)
            
            self.conv2 = nn.Conv2d(32,64,3,1,1)
            self.bn2 = nn.BatchNorm2d(64)

            self.conv3 = nn.Conv2d(64,128,3,1,1)
            self.bn3 = nn.BatchNorm2d(128)
            
            self.conv4 = nn.Conv2d(128,256,3,1,1)
            self.bn4 = nn.BatchNorm2d(256)

            self.conv5 = nn.Conv2d(256,512,3,1,1)
            self.bn5 = nn.BatchNorm2d(512)

            self.fc1 = nn.Linear(512*8*8,512)
            self.fc2 = nn.Linear(512,256)
            self.fc3 = nn.Linear(256,10)


      def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
            x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
            x = self.maxpool(self.relu(self.bn4(self.conv4(x))))
            x = self.maxpool(self.relu(self.bn5(self.conv5(x))))
            
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x
            
def init_model():
      model = Classifier()
      model.to(device)
      return model

def train_model(dataloader, model):
      optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
      loss_fn = nn.CrossEntropyLoss()

      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      total_loss  = 0

      for batch, (image,label) in enumerate(dataloader):
            image, label =  image.to(device), label.to(device)

            optimizer.zero_grad()
            pred = model(image)
            loss = loss_fn(pred,label)

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
      return 

def test_model(dataloader,model):
      loss_fn = nn.CrossEntropyLoss()
      correct_prediction = 0.0
      test_loss =0.0

      with torch.no_grad():
            for image, label in dataloader:
                  image, label =  image.to(device), label.to(device)
                  pred = model(image)

                  test_loss += loss_fn(pred,label).item()
                  
                  correct_prediction += (pred.argmax(1)==label).type(torch.float).sum().item()

      avg_loss = test_loss/len(dataloader)
      accuracy =(correct_prediction/len(dataloader.dataset))*100

      print(f"Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}\n")
      return

def train_test_model(train_dataloader, test_dataloader,model,epochs):
      start_time = time.time()
      for epoch in range(epochs):
            train_model(train_dataloader, model)
            test_model(test_dataloader,model)
      end_time = time.time()
      train_test_time = end_time - start_time
      print("Finished training")
      print(f"Total Training Time for {epochs} epochs: {train_test_time:.2f} seconds")
      print(f"Average Time per epochs: {train_test_time/epochs} seconds")




      