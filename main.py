
"""
Necessary libraries

pytorch
seaborn
matplotlib
numpy
pandas idk


"""

from CNN import dataloader
from CNN.dataloader import tomato_class_names
tomato_dataset = dataloader.load_tomato_dataset()
train_data,test_data,train_load,test_load = dataloader.create_train_and_test_dataset(tomato_dataset)

#images, labels = next(iter(train_load))
print(f"Number of tomato classes: {len(tomato_dataset.classes)}")
print(f"Class names: {tomato_dataset.classes}")
#print(f"Size of one batch (images): {images.shape}")
#print(f"Size of one batch (labels): {labels.shape}")
dataloader.show_random_sample(train_data,tomato_class_names)
