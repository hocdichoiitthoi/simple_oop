from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetProfiler:
    def __init__(self, output_dir: str | Path | None = None):
        self.df = None
        if output_dir is None:
            output_dir = Path(__file__).parent / "output"
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def load_csv(self, file_path: str | Path):
        file_path = Path(file_path)
        try:
            self.df = pd.read_csv(file_path)
            return self.df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV: {e}")
            return None

    def get_numerical_summary(self):
        if self.df is not None:
            return self.df.describe()
        return None

    def plot_and_save_histogram(self, column_name: str):
        if self.df is None:
            print("No data loaded.")
            return
        if column_name not in self.df.columns:
            print(f"Column not found. Available columns: {list(self.df.columns)}")
            return
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            print(f"Column '{column_name}' is not numeric.")
            return

        plt.figure(figsize=(8, 6))
        sns.histplot(self.df[column_name], kde=True)
        plt.title(f"Histogram of {column_name}")
        plt.xlabel(column_name)
        plt.ylabel("Frequency")

        filepath = Path(self.output_dir) / f"{column_name}_histogram.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        print(f"Saved histogram to {filepath}")
    def plot_and_save_boxplot(self, column_name: str):
        if self.df is None:
            print("No data loaded.")
            return
        if column_name not in self.df.columns:
            print(f"Column not found. Available columns: {list(self.df.columns)}")
            return
        if not pd.api.types.is_numeric_dtype(self.df[column_name]):
            print(f"Column '{column_name}' is not numeric.")
            return
        if "variety" not in self.df.columns:
            print("Column 'variety' not found; cannot create grouped boxplot.")
            return

        plt.figure(figsize=(15, 10))
        sns.boxplot(x="variety", y=column_name, data=self.df)
        plt.title(f"Boxplot of {column_name} by variety")
        plt.xlabel("Variety")
        plt.ylabel(column_name)

        filepath = Path(self.output_dir) / f"{column_name}_boxplot.png"
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
        print(f"Saved boxplot to {filepath}")
        
    def plot_and_save_pairplot(self):
        if self.df is None:
            print("No data loaded.")
            return
        if "variety" not in self.df.columns:
            print("Column 'variety' not found; pairplot requires a hue column.")
            return

        g = sns.pairplot(self.df, hue='variety')
        # pairplot returns a PairGrid; set the suptitle on its figure
        g.fig.suptitle('Pairplot of Iris Features by Variety', y=1.02)
        filepath = Path(self.output_dir) / "pairplot.png"
        g.fig.savefig(filepath, bbox_inches="tight")
        plt.close(g.fig)
        print(f"Saved pairplot to {filepath}")
        

if __name__ == "__main__":
    profiler = DatasetProfiler()
    data_file = Path(__file__).parent / "files" / "iris.csv"
    df = profiler.load_csv(data_file)
    if df is not None:
        print(df.head())
        summary = profiler.get_numerical_summary()
        if summary is not None:
            print("\nNumerical Summary:")
            print(summary)
        profiler.plot_and_save_histogram("sepal.length")
        profiler.plot_and_save_histogram("sepal.width")
        profiler.plot_and_save_histogram("petal.length")
        profiler.plot_and_save_histogram("petal.width")
        profiler.plot_and_save_boxplot("sepal.length")
        profiler.plot_and_save_boxplot("sepal.width")
        profiler.plot_and_save_boxplot("petal.length")
        profiler.plot_and_save_boxplot("petal.width")
        profiler.plot_and_save_pairplot()
        print("\nDone.")