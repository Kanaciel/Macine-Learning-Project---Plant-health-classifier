
"""
libraries to get i think
pytorch
seaborn
matplotlib
numpy
pandas idk
"""

from CNN import dataloader
from CNN.dataloader import tomato_class_names
from CNN import CNNmodel


tomato_dataset = dataloader.load_tomato_dataset()
train_data,test_data,train_load,test_load = dataloader.create_train_and_test_dataset(tomato_dataset)

class_to_idx = tomato_dataset.class_to_idx
dataset_class_names = [name for name, index in sorted(class_to_idx.items(), key=lambda item: item[1])]


#-----------datload test-----------#
'''
if tomato_class_names == dataset_class_names:
    print("Success: The manually defined 'tomato_class_names' list is correctly aligned with the dataset labels.\n")
else:
    print("Error: The 'tomato_class_names' list does not match the order of the dataset labels.\n")
    print("Manual List:", tomato_class_names)
    print("Dataset List:", dataset_class_names)
    

print(f"Number of tomato classes: {len(tomato_dataset.classes)}")
print(f"Class names: {tomato_dataset.classes}")
#dataloader.show_random_sample(train_data,tomato_class_names)

'''  

#----------model test ---------------#