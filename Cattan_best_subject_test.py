import moabb
from moabb.datasets import Cattan2019_VR
from mne import Epochs, find_events
from mne.decoding import Vectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from moabb.paradigms import P300

# Load the dataset
dataset = Cattan2019_VR()
paradigm = P300()

# Select a subject and load the raw data
subjects = range(1, 22)

results = []

for subject in subjects:
    X, y, _ = paradigm.get_data(dataset, subjects=[subject], return_epochs=True)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Create the classification pipeline
    pipeline = make_pipeline(
        Vectorizer(),  # Convert 3D data to 2D
        LogisticRegression(solver='liblinear')  # Use Logistic Regression
    )

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((subject, accuracy))

best_subject, best_score = max(results, key=lambda item: item[1])
print(f"\nBest performing subject: {best_subject:03d} with accuracy: {best_score:.3f}")