
import numpy as np

from util.activation_functions import Activation
from util.loss_functions import CrossEntropyError
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

        Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        self.layers = []
        self.layers.append(LogisticLayer(train.input.shape[1] - 1, 50, None, self.output_activation, True))
        # self.layers.append(LogisticLayer(50, 30, None, self.output_activation, True))
        self.layers.append(LogisticLayer(50, 10, None, self.output_activation, True))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        layer_inp = inp
        for layer in self.layers:
            layer.forward(layer_inp)
            layer_inp = np.insert(layer.outp, 0, 1)

    def _feed_backward(self, target):
        """
        Update the weights of the layers by propagating back the error
        """
        up_layer = None
        for layer in self.layers[::-1]:
            if up_layer is None: # output
                layer.deltas = target - layer.outp
            else:
                layer.computeDerivative(up_layer.deltas, up_layer.weights[1:,]) # removing bias weights
            up_layer = layer

        for layer in self.layers:
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        for img, label in zip(self.training_set.input, self.training_set.label):
            target = np.zeros(10, dtype=float)
            target[label] = 1.0
            self._feed_forward(img)
            self._feed_backward(target)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here

        # assume one hot encoding, if same probability for several classes,
        # class with smaller index is output
        self._feed_forward(test_instance)
        return np.argmax(self._get_output_layer().outp)

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
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
