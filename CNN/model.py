import torch
import torch.nn as nn



class Classifier(nn.Module):
      def __init__(self):
            super().__init__()
            #none distinct stuffs
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            
            #distinct stuffs
