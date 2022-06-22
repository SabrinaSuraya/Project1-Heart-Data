# Project1-Heart Data
Heart Disease Prediction
 Make prediction on patient have heart disease or not
 link: https://www.kaggle.com/code/ishanjat/heart-disease-prediction/data
 
 
 1. Summary
 - To make prediction on the patient have the heart disease or not
 - the deep learning model is used and trained
 - The model is self-made

2. IDE and Framework
- The project built with Spyder as the main IDE
- use Tensorflow, Keras, Numpy, Mathplot

3. Methodology
- The dataset was obtained in form of csv containing the 1024 smaples with 14 features.
- perform data cleaning to see the null data is available or not. In this project there is 
  no null data. we move on to another step
- Perform data preprocessing where we spilt data into feature(inputs) and label (output).     The output is in the form  of 1 or 0, to show that this project is binary classification     problem
- the model constist of 5 dense layers. 
- Model summary:
![image](https://user-images.githubusercontent.com/73817610/174966416-20240580-bc9e-4baf-b7b5-ded01dc06426.png)


-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= BinaryCrossentropy', metrics of accuracy, batch_size of 32 and epochs of 200
- The value is display by using TensorBoard:

![image](https://user-images.githubusercontent.com/73817610/174966276-b098a049-3961-4a8e-bb45-cb3f5e9ecd23.png)

![image](https://user-images.githubusercontent.com/73817610/174966190-0047117c-5bb8-4079-9b0a-cafb5e0093d9.png)

