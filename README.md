# Intro
Hiya this is a tomato plant classifier using machine learning

## What youre gonna need:
- Libraries
  - PyTorch
  - NumPy
  - Seaborn
  - Matplotlib
  - CustomTkinter
  - PIL

- Others
  - The data folder containing the datasets
  - User uploaded images can be anywhere
  - The models folder 

## Using the GUI
The GUI itself is relatively straightforward, just click the button and it will do what it says

- Train: click to train the model for 10  cycles, this cant be changed unfortunately
- Test: click to select an image and show what the model prediction is, there are some custom downloaded images in the data folder if you want, but any png/jg file should be fine
- Statistic: click to load the model's data, including its accuracy, F1-Score, and confusion matrix on the test dataset


## Special note
- The GUI will absolutely freeze up when training or trying to get the confusion matrix, as I have yet to implement threading.
- In the off chance the GUI is in light mode, the buttons are still there, you would have to hover above them for them to show however. 
- In case the code, doesnt run, you can download the exe compiled from [this drive here](https://drive.google.com/drive/folders/1NsvnUNYR3UrcMmfSho9Xzy8c686VX-xZ?usp=sharing)

