"""
libraries to get i think
pytorch
seaborn
matplotlib
numpy
pandas idk
customtkinter
pytorch gradcam
"""
import torch
from CNN import CNNdataloader,CNNmodel,CNNeval
from CNN.CNNdataloader import tomato_class_names
from GUI import GUIapp, GUItrain, GUItest, GUIeval






#-----------datload test-----------#
'''
tomato_dataset = CNNdataloader.load_tomato_dataset()
train_data,test_data,train_loader,test_loader = CNNdataloader.create_train_and_test_dataset(64,tomato_dataset)

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

#'''  


#----------  CLI test ---------------#

"""

tomato_dataset = CNNdataloader.load_tomato_dataset()
train_data,test_data,train_loader,test_loader = CNNdataloader.create_train_and_test_dataset(64,tomato_dataset)

classifier_model =CNNmodel.init_model("default model")

input_prompt =f"\nselect [1] to train model\nselect [2] to test model\n"
prompt =int(input(input_prompt))
if prompt ==1:    
    train_epochs = int(input("how many epochs"))
    CNNmodel.train_test_model(train_loader,test_loader,classifier_model,train_epochs, 0.0005)
    CNNmodel.save_model(classifier_model,"default model")
elif prompt == 2:
    accuracy = CNNeval.check_accuracy_from_dataset(classifier_model,test_data)
    print(f"accuracy of model is {accuracy:.2f}%")
    confusion_matrix = CNNeval.get_confusion_matrix(classifier_model,test_data,64)
    f1,precision, recall= CNNeval.get_F1_score(classifier_model,test_data,confusion_matrix,64)
    print(f"the F1 score is {f1}")
    print(f"the precision is {f1}")
    print(f"the recalls are {f1}")

    
    CNNeval.display_confusion_matrix(confusion_matrix)
    

"""


#------------------ GUI test -----------------------#

#"""

GUIapp.Init_App()


#"""