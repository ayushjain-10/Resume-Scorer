import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
 
train_df = pd.read_csv("C:\\Users\\brooklyn\\Downloads\\Train.csv")
test_df = pd.read_csv("C:\\Users\\brooklyn\\Downloads\\Test.csv")
final_scores_df = pd.read_csv("C:\\Users\\brooklyn\\Downloads\\Final_scores.csv")
 
train_df = train_df.drop(columns=["File name", "Awards/honours", "Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned"])
test_df = test_df.drop(columns=["File name", "Awards/honours", "Typos", "Chronological ordering", "Phrasing", "Relevancy of details mentioned"])
 
y_test = final_scores_df["Avg Format score"].values  
X_train = train_df.drop(columns=["Avg Format score"])
X_test = test_df.drop(columns=["Avg Format score"])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
isolation_forest = IsolationForest(contamination=0.05, random_state=42)
isolation_forest.fit(X_train_scaled)
anomaly_scores = isolation_forest.decision_function(X_train_scaled)
normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min()) * 5

print("Similarity Scores: ", normalized_scores)

y_pred = isolation_forest.predict(X_test_scaled)
y_pred_binary = [1 if pred == 1 else 0 for pred in y_pred]
 
threshold = 3.5
y_test_binary = (y_test > threshold).astype(int) 

print(y_test_binary)



