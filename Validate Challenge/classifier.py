import sklearn as sk
import pandas as pd

# First, we will clean the data a little bit, turning the binary
# flags that signify the labels into something more explicit.

data = pd.read_csv('patient_churn.csv')
data["label"] = ""

for index, row in data.iterrows():
    if(row["Beneficiary had a date of death prior to the start of the benchmark year"] == 1):
        row["label"] = "death"
    elif(row["Beneficiary identifier is missing"] == 1):
        row["label"] = "no_id"
    elif(row["Beneficiary had at least one month of Part A-only Or Part B-only Coverage"] == 1):
        row["label"] = "part-only"
    elif(row["Beneficiary had at least one month in a Medicare Health Plan"] == 1):
        row["label"] = "medicare"
    elif(row["Beneficiary does not reside in the United States"] == 1):
        row["label"] = "not-in-US"
    else:
        row["label"] = "other-initiatives"

# Here we remove the columns of the label flags.
data.drop(data.columns[[6, 7, 8, 9, 10, 11]], axis=1, inplace=True)

X = data.drop("label", axis=1)
Y = data["label"]

# Here we split the data into training and testing data at a 70/30 ratio, with a fixed random seed.
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(X, Y, test_size = 0.3, random_state = 0)

# To perform the multi-class classification, I chose to use an SVM.
# This is mostly due to personal experience and preference.
# We use the "one-vs-one" function shape for multiclass purposes.
SVM = sk.svm.SVC(decision_function_shape="ovo").fit(x_train, y_train)
results = SVM.predict(x_test)
print(SVM.score(results, y_test))