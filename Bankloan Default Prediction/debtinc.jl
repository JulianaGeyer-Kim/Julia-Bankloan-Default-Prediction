

#in Julia, do the folling--> using Pkg, Pkg.add("CSV"), Pkg.add("DataFrames"), Pkg.add("Statistics"), Pkg.add("MLJ"). Save to folder

using CSV
using DataFrames
using Statistics
using MLJ
import StatisticalMeasures.ConfusionMatrices as CM
using EvalMetrics, Random

# Load dataset
bankloan = CSV.read("bank-loan-dataset.csv", DataFrame)

# Define features and labels
X = select(bankloan, :debtinc)
y = bankloan.default
y = coerce(y, Multiclass)

# Split the data into training and testing sets
train, test = MLJ.partition(eachindex(y), 0.8, shuffle=true, rng=123)
X_train = X[train, :]
y_train = y[train]
X_test = X[test, :]
y_test = y[test]

# Train a Random Forest model
model = @load RandomForestClassifier verbosity=0 pkg=BetaML
forest = model()
rf = machine(forest, X_train, y_train)
MLJ.fit!(rf)

# Predict on the test set
y_hat = MLJ.predict(rf, X_test)
y_hat = MLJ.mode.(y_hat)  # Convert probabilities to classes

# Evaluate the model
accuracy = MLJ.accuracy(y_hat, y_test)
println("Accuracy: $accuracy")

# Confusion matrix
cm = CM.confmat(y_hat, y_test)

# Extract matrix from confusion matrix object
cm_ = StatisticalMeasures.ConfusionMatrices.matrix(cm)

# Precision, recall, and F1 for the TRUE class (assuming index 1 represents TRUE)
precision_TRUE = cm_[1,1] / sum(cm_[1, :])
recall_TRUE = cm_[1,1] / sum(cm_[:, 1])
f1_TRUE = 2 * precision_TRUE * recall_TRUE / (precision_TRUE + recall_TRUE)

# Precision, recall, and F1 for the FALSE class (assuming index 2 represents FALSE)
precision_FALSE = cm_[2,2] / sum(cm_[2, :])
recall_FALSE = cm_[2,2] / sum(cm_[:, 2])
f1_FALSE = 2 * precision_FALSE * recall_FALSE / (precision_FALSE + recall_FALSE)

# Print metrics
println("Precision TRUE: $precision_TRUE")
println("Precision FALSE: $precision_FALSE")
println("Recall TRUE: $recall_TRUE")
println("Recall FALSE: $recall_FALSE")
println("F1 TRUE: $f1_TRUE")
println("F1 FALSE: $f1_FALSE")


#=
VALUES AND QUALITY OF TRAINING PROCESS:
The accuracy value is .7428, which equates to being 74.28% accurate.
Highy reliable machine learning algorithm since there is a precision percentage of 83.1%
 and a failure rate of ____ 
=#





