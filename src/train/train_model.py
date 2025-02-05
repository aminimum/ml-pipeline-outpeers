import argparse
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc
import matplotlib.pyplot as plt


mlflow.autolog()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="params.yaml")
    return parser.parse_args()

def train_model(config):
    processed_path = "data/processed/processed.csv"
    df = pd.read_csv(processed_path)
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    test_size = config['train']['test_size']
    random_state = config['train']['random_state']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    model_params = config['train']['model_params']
    clf = RandomForestClassifier(**model_params)
    
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    with mlflow.start_run():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
        auc_score = auc(fpr, tpr)

        # Log AUC score
        mlflow.log_metric("AUC", auc_score)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        # Save the figure
        roc_curve_path = "roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.close()

        # Log ROC curve image in MLflow
        # mlflow.log_artifact(roc_curve_path)

        # mlflow.log_params(model_params)
        # mlflow.log_param("test_size", test_size)
        # mlflow.log_metric("accuracy", acc)
        # mlflow.log_metric("precision", prec)
                
        model_path = "models/model.pkl"
        joblib.dump(clf, model_path)
        # mlflow.sklearn.log_model(clf, "model")
        print(f"Model trained with accuracy: {acc}")

def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    train_model(config)

if __name__ == "__main__":
    main()
