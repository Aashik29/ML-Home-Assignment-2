# ML-Home-Assignment-2

CS5710 Machine Learning - Home Assignment 2

Student Information
Student Name: Aashik shaik

Student ID: 700758163

Assignment Overview
This assignment focuses on implementing, evaluating, and analyzing fundamental classification algorithms using the scikit-learn library in Python. The notebook covers three main tasks:

Building and evaluating Decision Tree classifiers with varying depths.

Visualizing the impact of the k parameter on the decision boundaries of a k-Nearest Neighbors (kNN) classifier.

Performing a comprehensive performance evaluation of a kNN model, including confusion matrix, classification report, and ROC/AUC analysis.

All tasks are performed on the well-known Iris dataset.

Setup & Dependencies
To run the code in the notebook, you need a Python environment with the following libraries installed:

numpy

matplotlib

scikit-learn

You can install these dependencies using pip:

pip install numpy matplotlib scikit-learn jupyter

How to Run
Clone this repository to your local machine.

Navigate to the repository's root directory.

Launch Jupyter Notebook or JupyterLab:

jupyter notebook

Open the Assignment_2_Programming.ipynb file to view and execute the code.

Part B: Programming Solutions
Q7. Build a Decision Tree
This section explores how the max_depth hyperparameter affects the performance of a DecisionTreeClassifier.

Implementation: A decision tree was trained on the Iris dataset for max_depth values of 1, 2, and 3.

Results: Training and test accuracies were reported for each depth.

max_depth=1: Exhibited clear signs of underfitting, with low accuracy on both training and test sets. The model was too simple to capture the data's complexity.

max_depth=2: Achieved high accuracy on both sets, indicating a good fit and strong generalization.

max_depth=3: Showed a slight increase in training accuracy while maintaining high test accuracy. This demonstrates how increasing model complexity can begin to overfit by memorizing the training data, which is signaled by a growing gap between training and test performance.

Q8. kNN Classification & Decision Boundaries
This section visualizes how the decision boundary of a KNeighborsClassifier changes with different values of k.

Implementation: kNN models were trained on the first two features of the Iris dataset (sepal length and width) for k values of 1, 3, 5, and 10.

Analysis: The decision boundaries were plotted for each k.

k=1: The boundary is highly irregular and complex, indicating a high-variance model that is overfitting the training data.

As k increases to 3, 5, and 10: The boundaries become progressively smoother. This illustrates the bias-variance tradeoff: as k grows, the model's variance decreases (less overfitting) but its bias increases (more generalized, potentially underfitting).

Q9. Performance Evaluation Programming
This section provides a comprehensive performance analysis of a kNN classifier with k=5.

Implementation: A kNN model was trained on the full Iris dataset and evaluated using multiple metrics from sklearn.metrics.

Evaluation Metrics:

Confusion Matrix: A 3x3 matrix was generated and plotted to visualize the model's performance for each of the three Iris species, showing correct and incorrect predictions.

Classification Report: A detailed report was printed, providing per-class precision, recall, and F1-score, along with the overall accuracy.

ROC Curve and AUC: Since the Iris dataset is multiclass, a One-vs-Rest (OvR) strategy was used to plot an ROC curve for each class. The Area Under the Curve (AUC) was calculated for each class, providing a robust measure of the model's ability to distinguish between classes.
