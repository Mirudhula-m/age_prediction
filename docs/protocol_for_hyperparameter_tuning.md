# Evaluation of different hyperparamters for vanilla-MLP
@author: Mirudhula Mukundan

### Evaluation is done based on the below conditions:

1) Convergence (how quickly they reach the answer)
2) Precision (how close is the prediction with respect to the actual)
3) Robustness (does it perform well for different set of training and testing set as well) -- perform this test at end, if possible
4) General overall performance (any other performance reviews that are not the above three)

The best way to do a hyperparameter tuning is to perform a [Grid Search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
The method for this and code can be referred to from this website [here](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/).
They have steps to perform Grid Search for each hyperparameter, so we can follow the same to reach a good set of parameters for our data too. For all your parts below, refer to the corresponding component in the link to see how to do the same.

Each hyperparameters to be tuned can be split amongst the three of us in the following way:

#### 1st round of hyperparamter tuning:

1) Jiayin: Grid Search for Optimization Algorithm (SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam)
2) Cathy: Grid Search for Network weight initialization (you can refer to the what is there in the link but also try to use the custom weights you initialized)
3) Mirudhula: Grid Search for Learning Rate and Momentum (learning rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]; and momentum = [0, 0.2, 0.4, 0.7, 0.9] )

Report your results of the tuning here:
1) Jiayin:
2) Cathy:
3) Mirudhula:

#### 2nd round of hyperparameter tuning:

1) Jiayin: Grid Search for Activation Function (Linear, ReLU, Leaky ReLU, tanh, sigmoid, hard sigmoid)
2) Cathy: Grid Search for Number of Neurons in the layer (one layer only - [50,100,1000,8000,17658,20000])
3) Mirudhula: Grid Search for Dropout rate (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

Report your results of the tuning here:
1) Jiayin:
2) Cathy:
3) Mirudhula:

#### Last round:

Number of layers in the network based on the above tuned parameters: to be decided.

Report your results of the tuning here:


### Tips for running the task:

1) Don't need to run these experiments on the full set of training data. You may take around 15 samples for training and 10 samples for testing. This hyperparameter tuning is just to see which setting is best for the complete data.
2) Once you decide on a final tuned parameter, consider running only the tuned parameter on the complete dataset or another unseen 15-10 train-test samples, to validate your results.
3) If you consider using the complete dataset for hyperparameter tuning, consider using GPUs and parallelization on multiple AWS consoles.

Let me know if there are any concerns.
