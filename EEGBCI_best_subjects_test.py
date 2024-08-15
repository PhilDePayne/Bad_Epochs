import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA
from mne.decoding import CSP
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Parameters
subjects = range(1, 110)  # EEG BCI dataset typically has subjects numbered 1-109
runs = [6, 10, 14]  # Motor imagery tasks

# Initialize a list to store results
results = []

for subject in subjects:
    try:
        # Load the data for the subject
        raw_fnames = eegbci.load_data(subject, runs)
        raw = mne.io.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])
        
        # Apply band-pass filter
        raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        
        # Extract events and epochs
        events, _ = mne.events_from_annotations(raw)
        event_id = dict(hands=2, feet=3)  # Event IDs for motor imagery tasks
        epochs = mne.Epochs(raw, events, event_id, tmin=0., tmax=4., baseline=None, preload=True)
        
        # Prepare the data for classification
        X = epochs.get_data()
        y = epochs.events[:, -1]
        
        # Set up the classifier with CSP (Common Spatial Patterns)
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        clf = make_pipeline(csp, LogisticRegression(max_iter=1000))
        
        # Evaluate using cross-validation
        scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
        mean_score = scores.mean()
        
        # Store the results
        results.append((subject, mean_score))
        print(f"Subject {subject:03d} | Classification accuracy: {mean_score:.3f}")
        
    except Exception as e:
        print(f"Subject {subject:03d} | Error: {str(e)}")
        continue

# Find the best-performing subject
best_subject, best_score = max(results, key=lambda item: item[1])
print(f"\nBest performing subject: {best_subject:03d} with accuracy: {best_score:.3f}")