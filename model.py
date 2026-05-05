import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN

def load_data():
    df = pd.read_csv("crime_india_dataset.csv")

    # clean columns
    df.columns = df.columns.str.strip()

    # datetime
    df['Date_Time'] = pd.to_datetime(df['Date_Time'])
    df['hour'] = df['Date_Time'].dt.hour
    df['month'] = df['Date_Time'].dt.month

    # encode crime type
    le = LabelEncoder()
    df['Crime_Type_Label'] = le.fit_transform(df['Crime_Type'])

    return df, le


def train_model(df):
    features = ['Latitude', 'Longitude', 'hour', 'month']
    X = df[features]
    y = df['Crime_Type_Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    return clf, scaler


def get_hotspots(df):
    coords = df[['Latitude', 'Longitude']]
    db = DBSCAN(eps=0.01, min_samples=5)
    df['cluster'] = db.fit_predict(coords)
    return df