import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("data/king_rook_vs_king.csv")

##preprocessing
columns = ["white_king_file", "white_rook_file", "black_king_file"]
file_map = dict(zip("abcdefgh", range(1,9)))
data[columns] = data[columns].replace(file_map)

##transforming target value
mapping = {"draw": 0, "zero": 1, "one": 1, "two": 1, "three": 1, "four": 1, "five": 2, "six": 2, "seven": 2, "eight": 2, "nine": 3, "ten": 3, "eleven": 3, "twelve": 3, "thirteen": 4, "fourteen": 4, "fifteen": 4, "sixteen": 4}

data["target"] = data["white_depth_of_win"].map(mapping)
data.drop(columns=["white_depth_of_win"], inplace=True)

data = data.drop_duplicates()

##model
X = data[["white_king_file", "white_king_rank","white_rook_file", "white_rook_rank","black_king_file", "black_king_rank"]]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier()

search_params = {
    "n_neighbors": [3, 5, 7, 9],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]}

tuner = GridSearchCV( 
    estimator=model,
    param_grid=search_params,
    scoring="f1_weighted",
    cv=5
)

tuner.fit(X_train_scaled, y_train)

best_model = tuner.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print(f"Best combination of hyperparameters:  {tuner.best_params_}")


##Evaluation
report = classification_report(y_test, y_pred)
print("Evaluation metrics: \n")
print(report)