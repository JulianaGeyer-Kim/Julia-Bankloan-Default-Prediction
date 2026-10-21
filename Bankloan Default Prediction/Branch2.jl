
X = select(bankloan, [:debtinc, :creddebt, :othdebt, :income])
y = bankloan.default
y = coerce(y, Multiclass)



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


y_hat = MLJ.predict(rf, X_test)
y_hat = MLJ.mode.(y_hat) 


accuracy = MLJ.accuracy(y_hat, y_test)
println("Accuracy: $accuracy")


cm = CM.confmat(y_hat, y_test)

cm_ = StatisticalMeasures.ConfusionMatrices.matrix(cm)

precision_TRUE = cm_[1,1] / sum(cm_[1, :])
recall_TRUE = cm_[1,1] / sum(cm_[:, 1])
f1_TRUE = 2 * precision_TRUE * recall_TRUE / (precision_TRUE + recall_TRUE)


println("Precision TRUE: $precision_TRUE")
println("Recall TRUE: $recall_TRUE")
println("F1 TRUE: $f1_TRUE")




