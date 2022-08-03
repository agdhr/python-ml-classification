# SUPPORT VECTOR MACHINE
# - https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python
# - https://stackabuse.com/implementing-svm-and-kernel-svm-with-pythons-scikit-learn/

# Import scikit-learn dataset library
from sklearn import datasets
# Load dataset
cancer = datasets.load_breast_cancer()
# Print the names of the 13 features
print("Features : ", cancer.feature_names)
# Print the label type of cancer
print("Labels : ", cancer.target_names)
# Print data shape
print(cancer.data.shape)
# print the cancer data features (top 5 records)
print(cancer.data[0:5])
# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)    # 70% training and 30% test

# Import svm model
from sklearn.svm import SVC
# Import skicit-learn metrics module for accuracy calculation
from sklearn.metrics import confusion_matrix,classification_report

# LINEAR KERNEL
clf_linear = SVC(kernel='linear')                       # Create a svm classifier
clf_linear.fit(X_train, y_train)                        # Train the model using the training sets
y_pred_linear = clf_linear.predict(X_test)              # Predict the response for test dataset
print(confusion_matrix(y_test, y_pred_linear))          # Confusion matrix
print(classification_report(y_test, y_pred_linear))     # Summary of classification report

# POLYNOMIAL KERNEL
clf_poly = SVC(kernel='poly', degree=8)
clf_poly.fit(X_train, y_train)
y_pred_poly = clf_poly.predict(X_test)
print(confusion_matrix(y_test, y_pred_poly))
print(classification_report(y_test, y_pred_poly))

# GAUSSIAN KERNEL
clf_gaus = SVC(kernel='rbf')
clf_gaus.fit(X_train, y_train)
y_pred_gaus = clf_gaus.predict(X_test)
print(confusion_matrix(y_test, y_pred_gaus))
print(classification_report(y_test, y_pred_gaus))

# SIGMOID KERNEL
clf_sigm = SVC(kernel='sigmoid')
clf_sigm.fit(X_train, y_train)
y_pred_sigm = clf_sigm.predict(X_test)
print(confusion_matrix(y_test, y_pred_sigm))
print(classification_report(y_test, y_pred_sigm))
