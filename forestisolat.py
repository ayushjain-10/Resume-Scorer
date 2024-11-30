import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
 
train_df = pd.read_csv("C:\\Users\\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\Train.csv")
test_df = pd.read_csv("C:\\Users\\Tanvi\\Pictures\\OneDrive\\Pictures\\College\\NEU Semester 1\\Foundations of AI\\Final Project\\Test.csv")
#final_scores_df = pd.read_csv("C:\\Users\\brooklyn\\Downloads\\Final_scores.csv")
 
train_df = train_df.drop(columns=["File name", "Awards/honours", "Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned"])
test_df = test_df.drop(columns=["File name", "Awards/honours", "Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned"])
 
#y_test = final_scores_df["Avg Format score"].values  
X_train = train_df.drop(columns=["Avg Format score"])
X_test = test_df.drop(columns=["Avg Format score"])
print(X_test.head())
X_test.loc[len(X_test)] = [5, 0, 5, 3.333]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(X_train_scaled)
anomaly_scores = isolation_forest.decision_function(X_test_scaled)
normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) * 5

# Closer to 0 is more "anomalous", closer to 5 is more "normal"
print("Similarity Scores: ", normalized_scores)
