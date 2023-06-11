# %%


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the dataset
data = pd.read_csv('E:heart_failure_clinical_records_dataset.csv')
data.head()

# %%
data['DEATH_EVENT'].value_counts()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

sns.displot(data['DEATH_EVENT'],bins=3,kde=False)
plt.title("ZeroR")
plt.xticks([0,1])
plt.show()

# %%
# Prepare the features and target variables
X = data.drop('anaemia', axis=1)
y = data['DEATH_EVENT']

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Initialize and train the ZeroR model
model = DummyClassifier(strategy="most_frequent")
model.fit(X_train, y_train)


# %%
# Make predictions on the testing set
y_pred = model.predict(X_test)

# %%
# Print the classifier output
print("Classifier Output:")
print(y_pred)

# %%
from sklearn.metrics import classification_report

# Assuming y_test and y_pred are the true labels and predicted labels, respectively
report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)


# %%
# Calculate and print the overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# %%
# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# %%

