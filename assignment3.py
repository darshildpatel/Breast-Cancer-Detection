import numpy as np
import copy


class KNN:
    def __init__(self, k):
        #KNN state here
        #Feel free to add methods
        self.k = k

    def nsmallest(self, k, arraytosort):
        final_list = []

        for i in range(k):
            min1 = max(arraytosort)
            for j in range(len(arraytosort)):
                if arraytosort[j] < min1:
                    min1 = arraytosort[j]

            arraytosort.remove(min1)
            final_list.append(min1)

        return final_list

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB)**2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        self.train = X
        self.labels = y

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        solution = []
        for testRow in X:
            dist_vector = [(self.distance(testRow, row), row) for row in self.train]
            best_dist = self.nsmallest(self.k, list(([row[0] for row in dist_vector])))
            best_labels = [vector[1] for vector in zip(dist_vector, self.labels) if vector[0][0] in best_dist]
            solution.append(np.bincount(np.array(best_labels)).argmax())
        return np.array(solution)


class decision_node:
    def __init__(self, attribute, valuesInA):
        self.attribute = attribute
        self.child = dict()#dict.fromkeys(valuesInA , None)

import math
class ID3:
    def __init__(self, nbins, data_range):
        #Decision tree state here
        #Feel free to add methods
        self.bin_size = nbins
        self.range = data_range

    def preprocess(self, data):
        #Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size*norm_data).astype(int)
        return categorical_data

    def entropy(self, n1, listOfLabels):
        n = list(n1)
        ent = 0
        for i in listOfLabels:
            ent += (n.count(i) / len(n)) * math.log((n.count(i) / len(n)), 2)
        return -ent

    def informationGain(self, attribute, decisionClass):
        attributewithclass = np.column_stack((attribute, decisionClass))
        parentEntropy = self.entropy(attributewithclass[:, [1]], np.unique(attributewithclass[:, [1]]))
        childEntropy = 0
        for i in np.unique(attribute):
            childDataSet = attributewithclass[attributewithclass[:, 0] == i][:, np.array([False, True])]
            childEntropy += (len(childDataSet) / len(attributewithclass)) * self.entropy(childDataSet, np.unique(childDataSet))
        return parentEntropy - childEntropy

    def plurality_value(self, examples):
        labels = list(examples[:, -1])
        uniquelabels = np.unique(labels)
        maxuniquelabels = [(i, labels.count(i)) for i in uniquelabels]
        best = sorted(maxuniquelabels, key=lambda x: x[1])[0][1]
        bestlabel = [pair[0] for pair in maxuniquelabels if pair[1] == best]
        return bestlabel[0]

    def importance(self, attributes, examples):
        result_gain = dict()
        for i in attributes:
            reqAttr = examples[:, i]
            label = examples[:, -1]
            result_gain[i] = self.informationGain(reqAttr, label)
        ordered_gain = sorted(result_gain.items(), key=lambda x: x[1], reverse=True)
        # print(ordered_gain)
        return ordered_gain[0][0]

    def decision_tree_learning(self, examples, attributes, parent_examples):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        elif (len(np.unique(np.array(examples)[:, -1]))) == 1:
            return np.unique(np.array(examples)[:, -1])[0]
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.importance(attributes, examples)
            # print("A->>",A)
            #             valuesInA = np.unique(examples[:, A])
            valuesInA = np.array([0, 1, 2, 3])
            decisionTree = {A: {}}
            #             tree = decision_node(A, valuesInA)
            attributes.remove(A)
            attr_send = copy.deepcopy(attributes)
            for value in valuesInA:
                exs = [e for e in examples if e[A] == value]
                #                 newAttr = copy.deepcopy(attributes)
                #                 newAttr.remove(A)
                subtree = self.decision_tree_learning(np.array(exs), attr_send, examples)
                #                 subtree = self.decision_tree_learning(np.array(exs),newAttr, examples)
                #                 tree.child[value] = subtree
                # print("Values -- >",value)
                decisionTree[A][value] = subtree
                # print("subtree -->",subtree)
                # print("decisiontree -->",decisionTree)

        return decisionTree

    def train(self, X, y):
        #training logic here
        #input is array of features and labels
        categorical_data = self.preprocess(X)
        tree = self.decision_tree_learning(np.hstack((categorical_data, y.reshape(categorical_data.shape[0], 1))), list(range(30)), [0])
        self.tree = tree
        # print(tree)
        return tree

    def predict(self, X):
        predictions = []
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        for test in categorical_data:
            copyOfTree = copy.deepcopy(self.tree)
            while copyOfTree != 1 or copyOfTree != 0:
                node = list(copyOfTree.keys())[0]
                branchValue = test[node]
                ranges = copyOfTree[node].keys()

                if branchValue not in ranges:
                    copyOfTree = np.random.randint(2)
                    predictions.append(copyOfTree)
                    break
                if isinstance(copyOfTree[node][branchValue], int):
                    copyOfTree = (copyOfTree[node][branchValue]).copy()
                else:
                    copyOfTree = (copyOfTree[node][branchValue]).copy()
                if copyOfTree == 0:
                    predictions.append(0)
                    break
                if copyOfTree == 1:
                    predictions.append(1)
                    break

        return np.array(predictions)

class Perceptron:
    def __init__(self, w, b, lr):
        #Perceptron state here, input initial weight matrix
        #Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def stepFunc(self, x, weight, bias):
        if sum(np.multiply(x, weight)) + bias > 0:
            return 1
        else:
            return -1

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        labels = []
        for i in y:
            if i == 0:
                labels.append(-1)
            else:
                labels.append(i)

        while steps:
            prediction = []
            for x, d in zip(X, labels):
                steps -= 1
                pred = self.stepFunc(x, self.w, self.b)
                prediction.append(pred)
                if pred != d:
                    self.w += (self.lr) * (np.multiply(d, x))
            if prediction == labels:
                break

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        solutions = []
        for i in X:
            pred = self.stepFunc(i, self.w, self.b)
            if pred == -1:
                pred = 0
            solutions.append(pred)
        # print(solutions)
        return np.array(solutions)

class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if(i == 0):
                X, y = self.shuffle(X,y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            #print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)

class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w #Each column represents all the weights going into an output node
        self.b = b
        self.comp = None
        self.input = None

    def forward(self, inpu):
        #Write forward pass here
        self.input = inpu
        self.comp = inpu.dot(self.w) + self.b
        return  self.comp

    def backward(self, gradients):
        #Write backward pass here
        deltaW = self.input.T.dot(gradients)
        deltaB = np.sum(gradients,axis=0)
        self.w -= self.lr*deltaW
        self.b -= self.lr*deltaB
        return gradients.dot(self.w.T)


class Sigmoid:

    def __init__(self):
        self.comp = 0.0

    def forward(self, input):
        # Write forward pass here
        #input = np.array(input, dtype='float128')
        self.comp = 1 / (1 + np.exp(-input))
        return self.comp

    def backward(self, gradients):
        # Write backward pass here
        back = (gradients) * (1 - self.comp) * (self.comp)
        return back