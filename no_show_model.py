import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Sample dataset
data = {
    'Age': [25, 40, 60, 30, 50, 70, 35, 55],
    'Gender': ['M', 'F', 'F', 'M', 'F', 'M', 'F', 'M'],
    'PreviousNoShows': [0, 2, 3, 1, 2, 4, 1, 3],
    'NoShow': [0, 1, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

df['Gender'] = LabelEncoder().fit_transform(df['Gender'])

X = df[['Age', 'Gender', 'PreviousNoShows']]
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print(classification_report(y_test, pred))