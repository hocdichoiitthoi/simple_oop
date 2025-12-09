# simple-example

A small Python example project demonstrating a DatasetProfiler class that loads a CSV, prints a numerical summary, and saves plots (histograms, boxplots, pairplot) to `output/`.

## Requirements
- Python 3.8+
- The project declares dependencies in `pyproject.toml` (pandas, matplotlib, seaborn).

## Install
From the project root (`D:\simple_example`) install in editable mode so dependencies are installed from `pyproject.toml`:

```powershell
pip install -e .
```

Alternatively, install dependencies manually:

```powershell
pip install pandas matplotlib seaborn
```

## Usage
Run the example script which will load `files/iris.csv`, print a short preview and summary, and save plots into `output/`:

```powershell
python basic_oop.py
```

Files produced are saved under `output/` (e.g. `sepal.length_histogram.png`, `pairplot.png`).

## Development
- Formatters / linters can be added to dev dependencies in `pyproject.toml`.

## License
Choose and add a license file (e.g., `LICENSE`) if you plan to publish the repository publicly.

---
If you want, I can commit this README and push it to your repository for you.
