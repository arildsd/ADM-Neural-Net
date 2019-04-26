import pandas
from sklearn.model_selection import train_test_split


df = pandas.read_csv(r"C:\Users\Oreo\PycharmProjects\ADM-Neural-Net\data\StudentsPerformance.csv")
labels = df[["math score","reading score","writing score"]]
features = df[["gender","race/ethnicity","parental level of education","lunch","test preparation course"]]
xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size=0.2, random_state=0)

xTrain[["math score","reading score","writing score"]] = yTrain
xTrain.to_csv("../data/student_performance_train.csv")

xTest[["math score","reading score","writing score"]] = yTest
xTest.to_csv("../data/student_performance_test.csv")