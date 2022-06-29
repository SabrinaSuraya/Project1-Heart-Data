# Project1-Heart Data
Heart Disease Prediction
 Make prediction on patient have heart disease or not.
 Do the Data Analysis - Descriptive statistics
 link: https://www.kaggle.com/code/ishanjat/heart-disease-prediction/data
 
 
 # 1. Summary
 - To make prediction on the patient have the heart disease or not
 - Do the data analysis
 - the deep learning model is used and trained
 - The model is self-made.

# 2. IDE and Framework
- The project built with Spyder as the main IDE
- use Tensorflow, Keras, Numpy, Mathplot

# 3. Methodology
- The dataset was obtained in form of csv containing the 1024 smaples with 14 features.
- perform data cleaning to see the null data is available or not. In this project there is 
  no null data. we move on to another step.
- Do the data analysis; correlation between the pair of feature
- Perform data preprocessing where we spilt data into feature(inputs) and label (output).     The output is in the form  of 1 or 0, to show that this project is binary classification     problem
- the model constist of 5 dense layers. 

# 4.Result

- Correlation:

![image](https://user-images.githubusercontent.com/73817610/176481193-acd7c8ec-4569-4de3-a6b8-d5e95ae09a30.png)

- histogram of age

![image](https://user-images.githubusercontent.com/73817610/176481321-7693fcc9-1d6c-4263-8a83-175ac8d0b2b1.png)

- histogram of target ( 1 or 0)

![image](https://user-images.githubusercontent.com/73817610/176481450-be44f208-f77f-496b-9a4c-40b3b27f0112.png)

- histogram of sex

![image](https://user-images.githubusercontent.com/73817610/176481648-1ae1dbf2-1e98-4ee9-9932-24de800890f3.png)

- density plot of age

![image](https://user-images.githubusercontent.com/73817610/176481772-f07a20de-4a90-4d51-bfef-d5f5a28c7f41.png)

- density plot of thalach

![image](https://user-images.githubusercontent.com/73817610/176481856-9a810639-03b5-4d85-ab3c-24ccca888c43.png)

- density plot of chol

![image](https://user-images.githubusercontent.com/73817610/176481926-cdde6ad4-d8f1-469d-9a91-e831938268bc.png)

- boxplot of data

![image](https://user-images.githubusercontent.com/73817610/176482049-cc62d7b0-ac7e-4fb6-bdc0-15f0da09d720.png)


# Data processing, Data training

use train test split

- Model summary:

![image](https://user-images.githubusercontent.com/73817610/174966416-20240580-bc9e-4baf-b7b5-ded01dc06426.png)

-The model is compile with optimizer of 'adam' with learning rate = 0.001, loss= BinaryCrossentropy', metrics of accuracy, batch_size of 32 and epochs of 200
- The value is display by using TensorBoard:

![image](https://user-images.githubusercontent.com/73817610/174966276-b098a049-3961-4a8e-bb45-cb3f5e9ecd23.png)

![image](https://user-images.githubusercontent.com/73817610/174966190-0047117c-5bb8-4079-9b0a-cafb5e0093d9.png)

# Evaluate Data

![image](https://user-images.githubusercontent.com/73817610/176488656-c869c321-6ede-4d6d-948a-138f2236aa72.png)

the top is for training and the bottom for test
- we can see that the model has 100% accuracy and almost 0 loss. The model is good and no need modifictaions


# Test Data
- make predictions
- make prediction on the first 5 test data

![image](https://user-images.githubusercontent.com/73817610/176482568-a7a02ce2-6042-4641-87bf-5326f981a520.png)

