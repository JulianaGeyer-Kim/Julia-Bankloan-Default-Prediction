# Bank Loan Default Prediction – Julia

A machine learning project in Julia that predicts whether a bank customer will default on a loan. It trains Random Forest classifiers on customer financial data and compares a **single-feature model** with a **four-feature model** using accuracy, precision, recall, and F1 score. Written for my Financial Technology course.

## Overview

Two scripts train the same type of model on different sets of features:

| Script | Features used | Evaluation |
|--------|---------------|------------|
| `[single-feature-file].jl` | `debtinc` | Accuracy, plus precision, recall, and F1 for both classes |
| `[four-feature-file].jl` | `debtinc`, `creddebt`, `othdebt`, `income` | Accuracy, plus precision, recall, and F1 for the TRUE class |

| Feature | Description |
|---------|-------------|
| `debtinc` | Debt-to-income ratio |
| `creddebt` | Credit card debt |
| `othdebt` | Other debt |
| `income` | Income |

The target is the `default` column. In both scripts, the data is split 80/20 into training and test sets.

## Methodology

1. **Load the data:** read `bank-loan-dataset.csv` into a DataFrame.
2. **Select features and target:** choose the input columns and use `default` as the label.
3. **Prepare the target:** convert `default` to a categorical (`Multiclass`) variable.
4. **Split the data:** 80% training and 20% testing, shuffled with a fixed seed (`rng=123`) so results are reproducible.
5. **Train the model:** fit a `RandomForestClassifier` from the BetaML package using MLJ.
6. **Predict:** generate predictions on the test set and take the most likely class for each row.
7. **Evaluate:** compute accuracy and a confusion matrix, then calculate precision, recall, and F1 score by hand from the matrix.

## Results

| Model | Accuracy |
|-------|----------|
| Single feature (`debtinc`) | 74.28% |
The four-feature script prints its own accuracy, precision, recall, and F1 score when run.

Precision, recall, and F1 for each class are printed when you run the scripts.

## Evaluation Metrics

- **Accuracy:** share of test rows classified correctly
- **Precision:** of the customers predicted to be in a class, how many actually were
- **Recall:** of the customers actually in a class, how many the model found
- **F1 score:** the harmonic mean of precision and recall

## Requirements

- Julia 1.6 or higher
- Packages: `CSV`, `DataFrames`, `Statistics`, `MLJ`, `BetaML`, `StatisticalMeasures`, `EvalMetrics`

Install them from the Julia REPL:

```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "MLJ", "BetaML", "StatisticalMeasures", "EvalMetrics"])
```

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/JulianaGeyer-Kim/your-repo-name.git
   cd your-repo-name
```
2. Place `bank-loan-dataset.csv` in the same folder as the scripts (see Dataset below).
3. Start Julia in that folder and run the single-feature script, which loads the data itself:
```julia
   include("[single-feature-file].jl")
```
4. To run the four-feature script, load the data first, because that script doesn't read the CSV itself:
```julia
   using CSV, DataFrames
   bankloan = CSV.read("bank-loan-dataset.csv", DataFrame)
   include("[four-feature-file].jl")
```

## Dataset

The scripts expect a file named `bank-loan-dataset.csv` with the columns `debtinc`, `creddebt`, `othdebt`, `income`, and `default`. The CSV data was provided by my instructor and is included in this repo.

## Skills Demonstrated

- Binary classification with a Random Forest
- Working with the MLJ machine learning framework in Julia
- Feature selection and model comparison
- Train/test splitting with a reproducible random seed
- Model evaluation with a confusion matrix
- Calculating precision, recall, and F1 score by hand

## Author

Juliana Geyer-Kim – [GitHub Profile](https://github.com/JulianaGeyer-Kim)
