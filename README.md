# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the SMS/email dataset containing messages and their labels (spam or ham).

2.Convert the text messages into numerical features using TF-IDF Vectorization.

3.Split the dataset into training and testing data.

4.Train an SVM (Support Vector Machine) classifier using the training data.

5.Predict whether new messages are Spam or Ham and evaluate the model using accuracy. 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Ahamed Hariz A
RegisterNumber:  212224060009
*/
```
~~~python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\spam.csv", encoding="latin-1")

# Select message and label columns
X = df["v2"]       # Message
y = df["v1"]       # spam / ham

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create SVM model
model = SVC(kernel="linear")

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Test a new message
message = ["Congratulations! You have won a free prize."]

message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)

print("\nMessage:", message[0])
print("Prediction:", prediction[0])
~~~
## Output:
<img width="994" height="310" alt="image" src="https://github.com/user-attachments/assets/7c1be26c-5bad-472b-ac4c-c20f19eb2c2b" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
