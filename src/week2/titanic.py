import os
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier

class TitanicPipeline:
    def __init__(self, output_dir: str | Path | None = None, random_state: int = 42):
        self.df = None
        self.random_state = random_state
        if output_dir is None:
            output_dir = Path(__file__).parent / "charts"
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, file_path: str | Path ) -> pd.DataFrame:
        file_path = Path(file_path)
        self.df = pd.read_csv(file_path)
        return self.df
        
    def clean_and_engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        cols = [
            "survived",
            "pclass",
            "sex",
            "age",
            "sibsp",
            "parch",
            "fare",
            "embarked",
            "cabin",
        ]
        df = df[cols]

        df["age"] = df["age"].fillna(df["age"].median())
        df = df.dropna(subset=["embarked", "fare", "sex", "pclass"]) 

        df["family_size"] = df["sibsp"].fillna(0) + df["parch"].fillna(0) + 1
        df["cabin_letter"] = df["cabin"].fillna("U").astype(str).str[0]

        bins = [0, 12, 18, 35, 60, 100]
        labels = ["child", "teen", "young_adult", "adult", "senior"]
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels)

        df = df.dropna(subset=["age_bin"]) 

        return df

    def prepare_split(self, df: pd.DataFrame):
        X = df[["pclass", "sex", "age_bin", "fare", "embarked", "cabin_letter", "family_size"]]
        y = df["survived"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        numeric_features = ["fare", "family_size"]
        numeric_transformer = StandardScaler()

        categorical_features = ["sex", "age_bin", "embarked", "cabin_letter"]
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="passthrough",
        )

        return X_train, X_test, y_train, y_test, preprocessor

    def train_and_evaluate(self, X_train, X_test, y_train, y_test, preprocessor):
    
        models = {
            "Logistic Regression": LogisticRegression(random_state=self.random_state),
            "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=500, min_samples_split=5, random_state=self.random_state),
            "XGBClassifier": XGBClassifier(n_estimators=6, eval_metric="logloss", random_state=self.random_state),
        }

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        results = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_proc, y_train)
            preds = model.predict(X_test_proc)
            acc = accuracy_score(y_test, preds)
            cm = confusion_matrix(y_test, preds)
            results[name] = {"accuracy": acc, "confusion_matrix": cm}

            print(f"{name} accuracy: {acc:.4f}")

            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix: {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            fname = os.path.join(self.output_dir, f"confusion_{name.replace(' ', '_').replace('/', '_')}.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

        return results

    def run(self):
        data_file = Path(__file__).parent / "file" / "titanic.csv"
        df = self.load_data(data_file)
        print(df.head())
        df = self.clean_and_engineer(df)
        print("---------------------------------------------")
        print("Data cleaned:")
        print(df.head())
        X_train, X_test, y_train, y_test, preprocessor = self.prepare_split(df)
        results = self.train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

        print("\nSummary:")
        for name, info in results.items():
            print(f"- {name}: accuracy={info['accuracy']:.4f}")

        return results


def main():
    pipeline = TitanicPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
