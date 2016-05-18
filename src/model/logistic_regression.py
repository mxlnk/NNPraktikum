# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np
from sklearn.metrics import accuracy_score

from model.logistic_layer import LogisticLayer
from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class LogisticRegression(Classifier):
    """
    A digit-7 recognizer based on logistic regression algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    learningRate : float
    epochs : positive int
    """

    def __init__(self, train, valid, test, learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        self.neuron = LogisticLayer(
            n_in = len(train.input[0]),
            n_out = 1,
            is_classifier_layer = True
        )

    def train(self, verbose=True):
        """Train the Logistic Regression.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch(verbose)

            if verbose:
                accuracy = accuracy_score(self.validationSet.label,
                                          self.evaluate(self.validationSet))
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy*100))
                print("-----------------------------")

    def _train_one_epoch(self, verbose=True):
        for instance, label in zip(self.trainingSet.input, self.trainingSet.label):
            biasedInput = np.append([1.0], instance)
            self.neuron.forward(biasedInput)
            self.neuron.computeDerivative(np.array([label]), None)
            self.neuron.updateWeights(self.learningRate)

    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """

        # Here you have to implement classification method given an instance
        biasedInput = np.append([1.0], testInstance)
        self.neuron.forward(biasedInput)
        return self.neuron.outp[0] > 0.5

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))
