# Machine-Learning-Tool

This app lets you train and test various supervised machine learning algorithms and visualize the accuracy (and r2 score) of their results. It allows you to adjust parameters that control the complexity of the algorithm model and save these results for future comparison. It also lets you upload and save your own custom datasets in CSV or Excel formats.

What is supervised machine learning?

Supervised machine learning involves providing input data (known as features or independent variables) along with an output variable (known as a label or dependent variable). The algorithm assesses each row of the data and generates a model that will attempt to label each row correctly.

How does supervised machine learning work?

Supervised machine learning consists of two phases: training and testing.

Training involves providing the model with both the input and output variables. The algorithm adjusts the model (also known as ‘fitting’) to find what combination of values of input variables will most often result in the provided outcome variable value.
In the testing phase, the algorithm is initially given only the input variables, makes a prediction of the outcome, and afterwards compares the result with the provided outcome label or value.

Once a model is trained and tested for a particular dataset, this app saves the accuracy (in binary classification, this is a percentage of correct vs. incorrect labels) or r2 score (in regression, this represents how well a model fits a dataset) for both the training and test phases, which can later be compared or plotted on a graph.

This app supports two types of machine learning algorithms: binary classification and regression.

What is binary classification?

Binary classification uses two possible values as an output or label (e.g., true or false, benign or malignant, spam or not spam). This app lets you choose many of the most common algorithms used in binary classification, including Logistic Regression, Decision Trees, and Feedforward Neural Networks.

What is regression?

Regression involves datasets with continuous numeric values (which can be integers or decimals) as the target or output value (e.g. the probability of gaining admission to a particular university, or the predicted price of a house). This app includes many common regression algorithms including Linear Regression, Ridge Regression, and Lasso Regression.

What is overfitting?

In supervised machine learning, there are trade-offs between the complexity of a model and the accuracy of the training and test results.

During the training phase, the more closely the model fits the training data, that is the more complex the model is, the higher the accuracy of the training results will be.

If the model fits the training data too well however, it will not score well during the test phase. This is called overfitting.

One way to avoid overfitting the training data is to simplify the model. Most machine learning models have various parameters that allow regularization (reducing the sensitivity of the model to the data at hand) or other forms of model simplification.

If a model is simplified, it may be less accurate during the training phase, but its accuracy will often improve during the test phase. This means it will generalize better, and thus it will be more effective in the real world.

Part of the task of applied machine learning is to find the best algorithm model and parameter values for a given dataset that generate the highest test score accuracy. This app helps the user accomplish this task without needing to code or understand the math behind the algorithms.
