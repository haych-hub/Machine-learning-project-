# Machine-learning-project-

MLEndClassification
Simple ML Classification Task using SVN, KNN, MLP, RFC, Naive Bayes Classifier, Logistic Regression. A deep MLP is also used for the second classification task

Basic Solution
Intonation Prediction

In this project I have built a model that predicts the intonation of a short audio segment. The audio dataset was prepared by the batch of MSc Big Data Science at QMUL.

The trainingMLEND.csv consits of 20k rows and 4 columns. Each row corresponds to one of the items in our dataset, and each item is described by four attributes.

File ID (audio file)
Numeral
Participand ID
Intonation
Here I have extracted 10 features from the audio signal namely

Power
Pitch mean
Pitch std
Voiced flag
Onset
MFCC
Zero Crossing Rate
Spectral Centroid
Spectral Rolloff
Root Mean Square
using librosa library.

I build 6 models which are

SVM
RandomForest
KNN
Naive Bayes
Logistic Regression
MuliLayer Perceptron Classifier
Grid Search is used to find the best parameters of each of the models. The models take the 10 features as input and try to classify them into the the 4 intonations.

Finally, Getting the highest Accuracy on the Random Forest Classifier model with the accuracy of 54.5% on the validation data with normalised predictors.

Advanced Solution
Next, I decided to develop a model that only identifies single digits as the accuracy achieved was much higher than that for a model that would classify the audio into all the numerical classes.

The Approach For Single Digit identification

The all 20000 audio files are iterated through and the 10 features are extracted as well as the 10 label classes.
The numpy array values are saved on the drive
NaN values are removed
The numpy arrays are converted to a dataframe for easy of encoding and preprocessing
The dataframe is transformed using a Standard Scaler
PCA is performed to extract 2 and 4 sets of vectors.
Data Visualized to analyse class distribution, correlation and relations between features.
Data is split between training and testing set
A Deep MLP using keras is trained
