# AIDI_2000_Applied_Machine_Learning
Repo to host all the assignments as well as projects of the course AIDI2000 - Applied Machine Learning

## Assignment #1: Titanic Survival Prediction
The goal of this assignment is to build a predictive model that determines whether a passenger survived the Titanic disaster based on various features.

In order to fulfill the goal of the task, a total of six ML models were utilized here. These models are inclusive of

1. Random Forest Classifier
2. Decision Tree Classifier
3. SVC (Support Vector Machine Classifier)
4. Logistic Regression
5. Gradient Boost Classifier
6. Artificial Neural Network (ANN)

For all the models a variety of performance metrics, namely,
1. Accuracy
2. Precision
3. Recall
4. F1-Score
5. Confusion Matrix
6. ROC Curves with AUC

were calculated and analyzed.

## Lab #1: 
### Task 1: Dataset selection
The Titanic Survival Prediction dataset was selected from all the available datasets, i.e., Iris dataset, Boston housing prices dataset, MINST dataset, wine quality dataset, & the Titanic dataset.
### Task 2: Data preprocessing
A variety of steps associated with data preprocessing were performed. These steps included but were not limited to
    1. performing of the data cleaning, like e.g., handling of the missing values, removing duplicates, etc.,
    2. spliting the data into training and testing sets.
### Task 3: Model Training and Evaluation
Gradient boosting & hist gradient boosting algorithms were the selected algorithms. Models were trained on the training dataset, evaluated using the model's performance using the testing dataset, and reported model's accuracy, precision, recall, and F1-score.
### Task 4: Model Tuning and Optimization
Performed some hyperparameter tuning using grid search & cross-validation to improve model performance.
### Task 5: Comparison & Conclusion
As the last step of the lab, a comparison of the tuned & untuned/ original model is presented, highlighting important findings.
	
## Lab #2: Building a Simple Two-Layer Neural Network & Experimenting with Model Hyperparameters
The goal of this lab is to understand how neural networks work. Furthermore, it provides a ground to experiment with model hyperparameters to understand their affects on the overall model performance.

The evaluated hyperparameters include
1. Number of neurons -> [2, 4, 10, 25]
2. Number of epochs -> ['relu', 'tanh', 'sigmoid']
3. The activation function -> [1000, 2500, 5000]
4. The model optimizer -> ['adam', 'sgd', 'rmsprop']

MLFlow has been exploited in this lab for model experiment tracking.