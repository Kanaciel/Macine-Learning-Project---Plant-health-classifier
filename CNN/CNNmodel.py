import torch
import torch.nn as nn

epochs = 1

class Classifier(nn.Module):
      def __init__(self):
            super().__init__()
            #none distinct stuffs
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(2,2)
            self.dropout = nn.Dropout1d(0.05)
            self.softmax =nn.Softmax()
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


      def foward(self, input):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.maxpool(self.relu(self.bn2(self.conv2(x))))
            x = self.maxpool(self.relu(self.bn3(self.conv3(x))))
            x = self.maxpool(self.relu(self.bn4(self.conv4(x))))
            x = self.maxpool(self.relu(self.bn5(self.conv5(x))))
            
            x = self.flatten(x)
            x = self.relu(self.fc1(x))
            x = self.dropout()
            x = self.relu(self.fc2(x))
            x = self.relu(self.fc3(x))

            
def init_model():
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model = Classifier()
      model.to(device)
      return model

def train_model(dataloader, model):
      optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)
      loss_fn = nn.CrossEntropyLoss()

