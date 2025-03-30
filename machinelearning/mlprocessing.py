
from abc import ABC, abstractmethod


class StatisticsResult:
    """
    Statistics for each class (how many from one element (ratio_element) are inside another union
    while removing elements)
    """
    # classes is a list/tupel of class-names for which we can then calculate its ratio to all the other classes
    def __init__(self, classes, counter):
        self.classes = classes
        self.counter = counter

    # ratio_element: the element we want to see (for example fibrosis to the rest)
    # remove_items: list of elements to be removed from the denominator/ratio calculation
    #               in our case - or quite usually - it is the background pixels
    def get_ratio(self, ratio_element, remove_items):
        nominator_value = 0
        denominator_value = 0
        for i in range(len(self.classes)):
            if self.classes[i] == ratio_element:
                nominator_value += self.counter[i]
            if self.classes[i] not in remove_items:
                denominator_value += self.counter[i]
        if denominator_value == 0:
            raise ZeroDivisionError
        return nominator_value / denominator_value


class MLProcessing(ABC):
    def __init__(self):
        self.images = []    # this list contains the image names
        self.marker_image = None
        self.markers = dict()
        self.parameters = dict()
        self.class_labels = []
        self.logfilename = "./phybrosoft.log"
        self.experiment_name = ""

    def set_experiment_name(self, name):
        self.experiment_name = name
        self.logfilename = "./phybrosoft_{}.log".format(name)

    def log(self, message):
        f = open(self.logfilename, "a+")
        f.write(message+"\n")
        f.close()

    def set_marker_image(self, imagefile):
        self.marker_image = imagefile

    def set_images(self, imagelist):
        self.images = imagelist

    def set_markers(self, markers_dict):
        self.markers = markers_dict

    def set_class_labels(self, classes):
        self.class_labels = classes

    def set_parameter(self, name, value):
        self.parameters[name] = value

    def _init_training(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def postprocessing(self, result):
        return result

    @abstractmethod
    def execute(self, imagefile):
        pass

    @abstractmethod
    def get_statistics_result(self) -> StatisticsResult:
        pass

    @abstractmethod
    def save_model(self, filename):
        pass

    @abstractmethod
    def load_model(self, filename):
        pass
