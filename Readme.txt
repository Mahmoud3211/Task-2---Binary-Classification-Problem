###Task 2 - Binary Classification Problem:

The Goal is to Create and train a machine learning model using the training set that performs well on the validation set,
so, first I started cleaning the training and validation data to be ready for classification, I started by reading the data 
correctly by taking into concederation that it is separated by semi colons (;) ,also separated the features or the independent variable 
from the target or dependent variable (classLabel), with that done i found that the data has many missing fields I choose the Forward fill
method to fill these missing data then performed one hot encoding technique to arrange the categorical variables , Finally there were columns 
with values of cordinatas or values separated with comma, I splitted thses values into to variables in the dataset I had the chance to scale the data but that 
did not improve the classification process.


Validation data had the same operation until i found thet one hot encoding the validation data would return a dataframe that does not have the same shape as the 
training data due to the fact that the validation data did not have some of the entries within the training data that was solved by creating a dataframe with columns 
that have the same names as the missing ones and joined this dataframe with the validation set.


Now the data is ready for classification techniques, I performed the following calssification techniques and the results were like the following:
Support Vector Classifiear model scored an acuuracy of: 54% 
AdaBoost Classifiear using DecisionTreeClassifier model scored an acuuracy of: 49.5%
Gaussian Naive Bayes Classifier model scored an acuuracy of: 65%
K Nearest Neighbors Classifier model scored an acuuracy of: 54.5%
After Tuning the Hyper-Parameters of K Nearest Neighbors Classifier model using GridSearch it model scored an acuuracy of: 57%
Stochastic Gradient Descent Classifier model scored an acuuracy of: 46.5%
Logistic Regression Classifier model scored an acuuracy of: 46.5%
After Tuning the Hyper-Parameters of the hyper-parameters of DecisionTreeClassifier using Gridsearch the model scored an acuuracy of: 49.5%
I created a Neural Network that works as a classifier and it scored an accuracy of : 46.5%

note: I tried tunning the parameters of SVC with Grid Search but it took long time with no results

The best accuracy was aquired by the Gaussian Naive Bayes Classifier model an it was : 65%

###Requirements:
These libraries must be pre-installed:
pandas
numby
matplotlib
sklearn
keras

###developed by :
Mahmoud Nada
