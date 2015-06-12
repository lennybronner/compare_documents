from __future__ import print_function

__author__ = 'lbronner'

from numpy import zeros, where, ravel, r_, single
from numpy.random import permutation
from scipy import ones
from pybrain.datasets import SupervisedDataSet, SequentialDataSet, ClassificationDataSet

class ExtendedClassificationDataSet(SupervisedDataSet):
    """ Specialized data set for classification data. Classes are to be numbered from 0 to nb_classes-1. """

    def __init__(self, inp, target=1, nb_classes=0, class_labels=None):
        """Initialize an empty dataset.
        `inp` is used to specify the dimensionality of the input. While the
        number of targets is given by implicitly by the training samples, it can
        also be set explicity by `nb_classes`. To give the classes names, supply
        an iterable of strings as `class_labels`."""
        # FIXME: hard to keep nClasses synchronized if appendLinked() etc. is used.
        SupervisedDataSet.__init__(self, inp, target)
        self.addField('class', 1)
        self.nClasses = nb_classes
        if len(self) > 0:
            # calculate class histogram, if we already have data
            self.calculateStatistics()
        self.convertField('target', int)
        if class_labels is None:
            self.class_labels = list(set(self.getField('target').flatten()))
        else:
            self.class_labels = class_labels
        # copy classes (may be changed into other representation)
        self.setField('class', self.getField('target'))

        self.addField("importance", nb_classes)
        self.link.append("importance")

    def addSample(self, inp, target, importance=None):
        if importance is None:
            importance = ones(len(target))
        self.appendLinked(inp, target, importance)

    def __add__(self, other):
        """Adds the patterns of two datasets, if dimensions and type match."""
        if type(self) != type(other):
            raise TypeError('DataSets to be added must agree in type')
        elif self.indim != other.indim:
            raise TypeError('DataSets to be added must agree in input dimensions')
        elif self.outdim != 1 or other.outdim != 1:
            raise TypeError('Cannot add DataSets in 1-of-k representation')
        elif self.nClasses != other.nClasses:
            raise IndexError('Number of classes does not agree')
        else:
            result = self.copy()
            for pat in other:
                result.addSample(*pat)
            result.assignClasses()
        return result

    def assignClasses(self):
        """Ensure that the class field is properly defined and nClasses is set.
        """
        if len(self['class']) < len(self['target']):
            if self.outdim > 1:
                raise IndexError('Classes and 1-of-k representation out of sync!')
            else:
                self.setField('class', self.getField('target').astype(int))

        if self.nClasses <= 0:
            flat_labels = list(ravel(self['class']))
            classes = list(set(flat_labels))
            self.nClasses = len(classes)

    def calculateStatistics(self):
        """Return a class histogram."""
        self.assignClasses()
        self.classHist = {}
        flat_labels = list(ravel(self['class']))
        for class_ in range(self.nClasses):
            self.classHist[class_] = flat_labels.count(class_)
        return self.classHist

    def getClass(self, idx):
        """Return the label of given class."""
        try:
            return self.class_labels[idx]
        except IndexError:
            print("error: classes not defined yet!")

    def _convertToOneOfMany(self, bounds=(0, 1)):
        """Converts the target classes to a 1-of-k representation, retaining the
        old targets as a field `class`.
        To supply specific bounds, set the `bounds` parameter, which consists of
        target values for non-membership and membership."""
        if self.outdim != 1:
            # we already have the correct representation (hopefully...)
            return
        if self.nClasses <= 0:
            self.calculateStatistics()
        oldtarg = self.getField('target')
        newtarg = zeros([len(self), self.nClasses], dtype='Int32') + bounds[0]
        for i in range(len(self)):
            newtarg[i, int(oldtarg[i])] = bounds[1]
        self.setField('target', newtarg)
        self.setField('class', oldtarg)
        # probably better not to link field, otherwise there may be confusion
        # if getLinked() is called?
        ##self.linkFields(self.link.append('class'))

    def _convertToClassNb(self):
        """The reverse of _convertToOneOfMany. Target field is overwritten."""
        newtarg = self.getField('class')
        self.setField('target', newtarg)

    def __reduce__(self):
        _, _, state, _lst, _dct = super(ClassificationDataSet, self).__reduce__()
        creator = self.__class__
        args = self.indim, self.outdim, self.nClasses, self.class_labels
        return creator, args, state, iter([]), iter({})

    def splitByClass(self, cls_select):
        """Produce two new datasets, the first one comprising only the class
        selected (0..nClasses-1), the second one containing the remaining
        samples."""
        leftIndices, dummy = where(self['class'] == cls_select)
        rightIndices, dummy = where(self['class'] != cls_select)
        leftDs = self.copy()
        leftDs.clear()
        rightDs = leftDs.copy()
        # check which fields to split
        splitThis = []
        for f in ['input', 'target', 'class', 'importance', 'aux']:
            if self.hasField(f):
                splitThis.append(f)
        # need to synchronize input, target, and class fields
        for field in splitThis:
            leftDs.setField(field, self[field][leftIndices, :])
            leftDs.endmarker[field] = len(leftIndices)
            rightDs.setField(field, self[field][rightIndices, :])
            rightDs.endmarker[field] = len(rightIndices)
        leftDs.assignClasses()
        rightDs.assignClasses()
        return leftDs, rightDs

    def castToRegression(self, values):
        """Converts data set into a SupervisedDataSet for regression. Classes
        are used as indices into the value array given."""
        regDs = SupervisedDataSet(self.indim, 1)
        fields = self.getFieldNames()
        fields.remove('target')
        for f in fields:
            regDs.setField(f, self[f])
        regDs.setField('target', values[self['class'].astype(int)])
        return regDs

if __name__ == "__main__":
    dataset = ClassificationDataSet(2, 1, class_labels=['Urd', 'Verdandi', 'Skuld'])
    dataset.appendLinked([ 0.1, 0.5 ]   , [0])
    dataset.appendLinked([ 1.2, 1.2 ]   , [1])
    dataset.appendLinked([ 1.4, 1.6 ]   , [1])
    dataset.appendLinked([ 1.6, 1.8 ]   , [1])
    dataset.appendLinked([ 0.10, 0.80 ] , [2])
    dataset.appendLinked([ 0.20, 0.90 ] , [2])

    dataset.calculateStatistics()
    print(("class histogram:", dataset.classHist))
    print(("# of classes:", dataset.nClasses))
    print(("class 1 is: ", dataset.getClass(1)))
    print(("targets: ", dataset.getField('target')))
    dataset._convertToOneOfMany(bounds=[0, 1])
    print("converted targets: ")
    print((dataset.getField('target')))
    dataset._convertToClassNb()
    print(("reconverted to original:", dataset.getField('target')))