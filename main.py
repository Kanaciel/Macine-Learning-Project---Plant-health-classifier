"""
libraries to get i think
pytorch
seaborn
matplotlib
numpy
pandas idk
customtkinter
pygame audio
pytorch gradcam
"""
import torch
from CNN import CNNdataloader
from CNN.CNNdataloader import tomato_class_names
from CNN import CNNmodel
from CNN import CNNeval


tomato_dataset = CNNdataloader.load_tomato_dataset()
train_data,test_data,train_loader,test_loader = CNNdataloader.create_train_and_test_dataset(64,tomato_dataset)



#-----------datload test-----------#
'''
class_to_idx = tomato_dataset.class_to_idx
dataset_class_names = [name for name, index in sorted(class_to_idx.items(), key=lambda item: item[1])]


if tomato_class_names == dataset_class_names:
    print("Success: The manually defined 'tomato_class_names' list is correctly aligned with the dataset labels.\n")
else:
    print("Error: The 'tomato_class_names' list does not match the order of the dataset labels.\n")
    print("Manual List:", tomato_class_names)
    print("Dataset List:", dataset_class_names)
    

print(f"Number of tomato classes: {len(tomato_dataset.classes)}")
print(f"Class names: {tomato_dataset.classes}")
#CNNdataloader.show_random_sample(train_data,tomato_class_names)

'''  

#----------model test ---------------#

classifier_model =CNNmodel.init_model("default model")

input_prompt =f"\nselect [1] to train model\nselect [2] to test the model with random image\n"
prompt =int(input(input_prompt))
if prompt ==1:    
    CNNmodel.train_test_model(train_loader,test_loader,classifier_model,20, 0.0005)
    CNNmodel.save_model(classifier_model,"default model")
elif prompt == 2:
    accuracy = CNNeval.check_accuracy_from_dataset(classifier_model,test_data)
    print(f"accuracy of model is {accuracy:.2f}%")

    