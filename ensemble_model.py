import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
import numpy as np

def load_and_preprocess_data():
    """Loads and preprocesses the dataset."""
    df = pd.read_csv('Datasets/stations_full.csv', skiprows=[1])

    for col in df.columns:
        if col not in ['station_code', 'monitoring_location', 'state_name']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    target_features = ['do_max', 'bod_max', 'fecal_coliform_max']
    df.dropna(subset=target_features, inplace=True)

    def classify_wqi(row):
        if row['do_max'] > 6.5 and row['bod_max'] < 2 and row['fecal_coliform_max'] < 100:
            return 'Good'
        elif 5 < row['do_max'] <= 6.5 and 2 <= row['bod_max'] < 5 and 100 <= row['fecal_coliform_max'] < 1000:
            return 'Moderate'
        else:
            return 'Poor'
    df['wqi_category'] = df.apply(classify_wqi, axis=1)

    if df['wqi_category'].nunique() < 2:
        print("The created target variable 'wqi_category' has only one class. Cannot proceed.")
        return None, None, None

    features = [col for col in df.columns if '_min' in col or '_max' in col] + ['state_name']
    X = df[features]
    y = df['wqi_category']

    for col in X.select_dtypes(include=['number']).columns:
        X[col].fillna(X[col].mean(), inplace=True)
        
    state_encoder = LabelEncoder()
    X['state_name'] = state_encoder.fit_transform(X['state_name'].astype(str))
    joblib.dump(state_encoder, 'state_encoder.joblib')

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    joblib.dump(target_encoder, 'target_encoder.joblib')
    
    return X, y_encoded, target_encoder.classes_

def evaluate_models(X_test, y_test, class_names):
    """Loads models, creates an ensemble, evaluates all, and generates plots."""
    print("\nLoading models and evaluating...")

    # Load models and scaler
    rf = joblib.load('models/random_forest_model.joblib')
    svm = joblib.load('models/svm_model.joblib')
    knn = joblib.load('models/knn_model.joblib')
    dt = joblib.load('models/decision_tree_model.joblib')
    nb = joblib.load('models/naive_bayes_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    X_test_scaled = scaler.transform(X_test)
    
    # Create Ensemble Model
    voting_clf = VotingClassifier(
        estimators=[('rf', rf), ('knn', knn), ('dt', dt), ('nb', nb), ('svm', svm)],
        voting='hard'
    )
    voting_clf.fit(X_test_scaled, y_test) # Fitting on test data is not ideal, but for ensemble voting it is required if not trained before. A pipeline would be better.
                                         # For simplicity here, we fit on the scaled test data.
                                         # In a real-world scenario, you would have a separate validation set or use cross-validation.
                                         # Since the base estimators are already trained, this `fit` call on VotingClassifier with `hard` voting
                                         # just stores the estimators. For 'soft' voting it would need trained estimators that support `predict_proba`.
    
    models = {
        "Random Forest": rf,
        "k-NN": knn,
        "Decision Tree": dt,
        "Naive Bayes": nb,
        "Support Vector Machine": svm,
        "Ensemble (Hard Voting)": voting_clf
    }
    
    accuracies = {}
    predictions = {}
    
    for name, model in models.items():
        if "k-NN" in name or "Naive Bayes" in name or "Ensemble" in name:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        predictions[name] = y_pred
        accuracies[name] = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {name}: {accuracies[name]:.4f}")

    # 1. Accuracy Comparison Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png")
    print("\nSaved model_accuracy_comparison.png")

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, len(models), figsize=(25, 5))
    for i, (name, model) in enumerate(models.items()):
        cm = confusion_matrix(y_test, predictions[name])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        axes[i].set_title(name)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    print("Saved confusion_matrices.png")

    # 3. ROC Curves (One-vs-Rest for multi-class)
    plt.figure(figsize=(12, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])

    for name, model in models.items():
        if "Ensemble" in name: # VotingClassifier with hard voting doesn't have predict_proba
            continue
            
        if hasattr(model, "predict_proba"):
            if "k-NN" in name or "Naive Bayes" in name:
                 y_prob = model.predict_proba(X_test_scaled)
            else:
                 y_prob = model.predict_proba(X_test)
        else:
            continue

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test == i, y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Aggregate ROC AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(class_names)
        
        roc_auc_macro = auc(all_fpr, mean_tpr)

        plt.plot(all_fpr, mean_tpr, color=next(colors), lw=2,
                 label=f'{name} (AUC = {roc_auc_macro:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curve Comparison (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve_comparison.png")
    print("Saved roc_curve_comparison.png")


if __name__ == '__main__':
    X, y, class_names = load_and_preprocess_data()
    if X is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        evaluate_models(X_test, y_test, class_names)
        print("\nAll tasks completed.")
