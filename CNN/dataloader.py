from pathlib import Path
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

#------------variables--------------#
#oh the woes of file management
root_path = Path(__file__).resolve().parent.parent
plantvillage_data_path = root_path/"Data"/"Tomato Leaves"
BATCH_SIZE = 32

tomato_class_names = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Spider mites Two-spotted spider mite",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "healthy"
]


trans_img_to_tensor = transforms.Compose([
      transforms.Resize((256,256)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
])

#------functions------#      
def load_tomato_dataset():
      tomato_dataset = datasets.ImageFolder(root=plantvillage_data_path, transform = trans_img_to_tensor)
      return tomato_dataset   

def create_train_and_test_dataset(tomato_dataset=None):
      if tomato_dataset is None:
            tomato_dataset = load_tomato_dataset()
      train_size = int(0.8* len(tomato_dataset))
      test_size = len(tomato_dataset) - train_size

      train_dataset, test_dataset = torch.utils.data.random_split(tomato_dataset, [train_size, test_size])

      train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle= False)

      return train_dataset,test_dataset,train_loader,test_loader

#display random img from dataset
#args are dataset and list of class name
def show_random_sample(dataset, class_names=None):
    idx = random.randint(0, len(dataset) - 1)
    img, label = dataset[idx]

    # Convert tensor (C,H,W) → (H,W,C)
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0)

    plt.imshow(img)
    plt.axis("off")

    if class_names is not None:
        plt.title(f"Label: {class_names[label]}")
    else:
        plt.title(f"Label: {label}")

    plt.show()

#-------------testing---------------#
