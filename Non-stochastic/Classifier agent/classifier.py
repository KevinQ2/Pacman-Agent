# classifier.py
# Lin Li/26-dec-2021
#
# Use the skeleton below for the classifier and insert your code here.
import math
import random as rnd

# A random forest classifier
class Classifier:
    def __init__(self):
        pass

    def reset(self):
        pass
    
    def fit(self, data, target):
        self.trees = self.generateForest(data, target, 100)

    def predict(self, data, legal=None):
        # Majority voting of decision trees
        votes = {}

        for tree in self.trees:
            vote = tree.predict(data)

            if vote in votes:
                votes[vote] += 1
            else:
                votes[vote] = 1

        bestVote = None
        bestCount = 0

        for vote in votes:
            if votes[vote] > bestCount:
                bestVote = vote
        
        return bestVote

    def sampleRecords(self, data, target, size):
        # Sample a fraction of the dataset
        nRecords = len(data)
        nSampleRecords = math.ceil(nRecords * size)

        indexes = range(nRecords)
        sampleIndexes = rnd.sample(indexes, nSampleRecords)

        x, y = [], []

        for i in sampleIndexes:
            x.append(data[i])
            y.append(target[i])
        
        return x, y

    def generateForest(self, data, target, nTrees):
        trees = []
        splittableFeatures = []

        # Get features worth splitting (has found more than 1 different values)
        for feature in range(len(data[0])):
            values = []

            for record in data:
                if record[feature] not in values:
                    values.append(record[feature])

                    if len(values) > 1:
                        splittableFeatures.append(feature)
                        break

        nFeatures = math.floor(math.sqrt(len(splittableFeatures)))

        # Create decision trees using subsets of data, and subsets of splittable features
        for i in range(nTrees):
            x, y = self.sampleRecords(data, target, 0.8)
            validFeatures = rnd.sample(range(len(splittableFeatures)), nFeatures)

            tree = DecisionTreeClassifier()
            tree.fit(x, y, validFeatures)
            trees.append(tree)
        
        return trees

class DecisionTreeClassifier:
    def __init__(self):
        pass
    
    def fit(self, data, target, validFeatures):
        self.classes = list(set(target))
        self.validFeatures = validFeatures

        # Create an empty class counter
        self.emptySamples = []

        for i in range(len(self.classes)):
            self.emptySamples.append(0)

        self.tree = self.generateTree(data, target, 0)

    def predict(self, data, legal=None):
        node = self.tree

        while node.featureIndex != -1:
            featureValue = str(data[node.featureIndex])

            # Choose current prediction if value has not appeared in the training data
            if featureValue not in node.branches:
                return node.predictedClass

            node = node.branches[featureValue]
        
        return node.predictedClass

    def bestSplit(self, data, target):
        totalSamples = len(target)
        
        currentClassSamples = list(self.emptySamples)
        for i in target:
            currentClassSamples[self.getClassIndex(i)] += 1

        # Find best feature
        bestFeature = None
        bestGini = None
        bestValues = None
        
        for feature in self.validFeatures:
            # Get number of samples per class per feature's value
            values = {}     # {value1: [class1, class2]}

            for record in range(len(data)):
                value = str(data[record][feature])

                if value not in values:
                    values[value] = list(self.emptySamples)

                values[value][self.getClassIndex(target[record])] += 1

            # Update best feature to split
            gini = 0

            if len(values) > 1: # No point selecting a feature that does not split
                for featureValue in values:
                    gini += (sum(values[featureValue]) / totalSamples) * self.calcGini(values[featureValue], totalSamples)

                if bestFeature == None:
                    bestFeature = feature
                    bestGini = gini
                    bestValues = list(values)
                elif gini < bestGini:
                    bestFeature = feature
                    bestGini = gini
                    bestValues = list(values)

        return bestFeature, bestValues

    def generateTree(self, data, target, depth):
        classSamples = list(self.emptySamples)
        for i in target:
            classSamples[self.getClassIndex(i)] += 1
        
        predictedClassIndex = classSamples.index(max(classSamples))
        node = Node(classSamples, self.classes[predictedClassIndex]) 

        # Return if all samples belong to the same class
        if node.gini == 0:
            return node
        
        # Create branches
        feature, values = self.bestSplit(data, target)
        
        if feature != None:
            node.featureIndex = feature
            branches = {}   # {branchValue: [data, targets]}

            # Split dataset into the relevant branches
            for value in values:
                branches[value] = [[], []]

            for record in range(len(data)):
                branchValue = str(data[record][feature])
                branches[branchValue][0].append(data[record])
                branches[branchValue][1].append(target[record])

            for branch in branches:
                node.branches[branch] = self.generateTree(branches[branch][0], branches[branch][1], depth + 1)

        return node

    def calcGini(self, classSamples, totalSamples):
        acc = 1

        for sample in classSamples:
            acc -= (sample / totalSamples) ** 2
        
        return acc

    def getClassIndex(self, targetValue):
        # Guaranteed to find index, since classes are pulled from the same dataset
        for i in self.classes:
            if i == targetValue:
                return i

class Node:
    def __init__(self, classSamples, predictedClass):
        self.samples = sum(classSamples)
        self.values = classSamples
        self.gini = self.calcGini()

        self.featureIndex = -1
        self.branches = {}

        self.predictedClass = predictedClass

    def calcGini(self):
        acc = 1

        for classSample in self.values:
            acc -= (classSample/self.samples) ** 2
        
        return acc
