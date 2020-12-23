import pandas as pd
import numpy as np


class gnb:
    """Implementation of a Gaussian Naive Bayes classifier built on top of pandas.

    To use this class, first create an instance, fit the model, then use the predict function to classify additional data.
    ```Python
    classifier = gnb()
    data = pd.DataFrame({
        'gender': ['male', 'male', 'male', 'male', 'female', 'female', 'female', 'female'],
        'height': [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75],
        'weight': [180, 190, 170, 165, 100, 150, 130, 150],
        'foot_size': [12, 11, 12, 10, 6, 8, 7, 9]
    })
    x = data[['height', 'weight', 'foot_size']]
    y = data['gender']
    classifier.fit(x, y)
    x_test = pd.Series({'height': 6, 'weight': 130, 'foot_size': 8})
    classifier.predict(x_test)
    ```
    """

    def __init__(self, mean=None, variance=None):
        self.mean = mean
        self.variance = variance

    def fit(self, x: pd.DataFrame, y: pd.Series):
        """Tain model on with x as the features and y as the class lables.

        Parameters
        ----------
        x : pd.DataFrame
            A labled data frame with the features used to predict the class.
        y : pd.Series
            Class labels for the provided data.
        """
        self.mean = x.groupby(y).mean()
        self.variance = x.groupby(y).var()

    def predict(self, observation: pd.Series):
        """Use the formula for Gaussian Naive Bayes to classify an observation

        Parameters
        ----------
        observation : pd.Series
            An observation vector with the same length as the training feature set.

        Returns
        -------
        Index
            Label of the class with the highest posterier probability
        """
        probs = (2*np.pi*self.variance)**-0.5 * \
            np.e**(-1*(observation-self.mean)**2/(2*self.variance))
        class_prob = 1/self.mean.shape[0]
        probs['temp'] = probs.apply(
            lambda x: np.prod(x.values)*class_prob, axis=1)
        probs['post'] = probs.apply(
            lambda x: x['temp']/np.sum(probs['temp'].values), axis=1)
        return probs['post'].idxmax()
