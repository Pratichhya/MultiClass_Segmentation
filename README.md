## MultiClass Segmentation
As a part of a research work done with Prof. Stefan Lang from University of Salzburg, here we tried to perform hyperparamter sensitivity when dealing with multi class dataset.
By Pratichhya Sharma

The entire script can be run by: python main.py

### Notebook:
1. Documentation_T1.ipnyb: It provides an overview of the obtained result
2. npy_generator_T1.ipnyb: It provides scripts to prepare npy file by performing basic preprocessing steps. Path to that npy file can then be updated in data_loader.py
which will then show the path to the dataset.

### Remarks:
- Data is not provided in this repository as it a private dataset was used for research purpose only. You can use any satellite image or even computer vision dataset that has 3 bands.
Or evn higher band dataset can be used with slight updating of the input layer to the model
- Incase of class with more than or less than 8 class as in this case please update the number of output channels in the models.

