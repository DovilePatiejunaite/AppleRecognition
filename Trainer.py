#!/usr/bin/env python
from optparse import OptionParser
from SimpleCV import *


class Trainer:

    def __init__(self):
        self.classifier = None
        self.options = None

    def set_classifier(self, classifier):
            self.classifier = classifier

    def get_classifier(self):
            return self.classifier

    @staticmethod
    def create_extractor(extractor_name):
        if extractor_name == 'hue':
            extractor = HueHistogramFeatureExtractor(10)
        elif extractor_name == 'edge':
            extractor = EdgeHistogramFeatureExtractor(10)
        elif extractor_name == 'haar':
            extractor = HaarLikeFeatureExtractor(fname='haar.txt')
        return extractor

    @staticmethod
    def create_classifier(classifier_name, extractors):
        if classifier_name == 'svm':
            classifier = SVMClassifier(extractors)
        elif classifier_name == 'tree':
            classifier = TreeClassifier(extractors)
        elif classifier_name == 'bayes':
            classifier = NaiveBayesClassifier(extractors)
        elif classifier_name == 'knn':
            classifier = KNNClassifier(extractors)
        return classifier

    @staticmethod
    def get_class_name_from_path(training_path):
        classes = []
        directory_list = os.listdir(training_path)
        for directory_name in directory_list:
            if os.path.isdir(training_path + '/' + directory_name):
                classes.append(directory_name)
        return classes

    @staticmethod
    def save_results(classifier, images, result_path):
        num = 1
        for img in images:
            class_name = classifier.classify(img)
            img.drawText(class_name, 10, 10, fontsize=20, color=Color.BLUE)
            img.save(result_path + '/' + 'result_%02d.jpg' % num)
            num += 1

    def train_classifier(self, classes, training_root_path):
        training_paths = [training_root_path + '/' + c for c in classes]
        self.classifier.train(training_paths,classes, verbose=False, savedata='features.tab')

    def test_classifier(self, classes, testing_root_path):
        testing_paths = [testing_root_path + '/' + c for c in classes]
        print "Test results", self.classifier.test(testing_paths, classes, verbose=False), "\n"

    def save_classifier(self, classifier_file_name):
        self.classifier.save(classifier_file_name)

    def get_class_names(self):
        return self.classifier.mClassNames

    def classify_image(self, image_file_name):
        image = Image(image_file_name)
        return self.classifier.classify(image)

    def parse_options(self, args):
        """
        Parse command-line options
        """
        parser = OptionParser()
        parser.add_option("-a", "--training", action="store", dest="training_path", default="training",
                          help="training samples path"),
        parser.add_option("-t", "--testing", action="store", dest="test_path", default="testing",
                          help="testing samples path"),
        parser.add_option("-r", "--results", action="store", dest="result_path", default="results",
                          help="testing results path"),
        parser.add_option("-c", "--classifier", action="store", dest="classifier_name", default="tree",
                          help="using classifier (svm|tree|bayes|knn)"),
        parser.add_option("-d", "--debug", action="store", dest="debug", default="False",
                          help="to test classifier or not"),
        (self.options, args) = parser.parse_args(args)

        if not self.options.classifier_name:
            parser.print_help()
            exit(0)


def process():
    # Init trainer and parse options from argv
    trainer = Trainer()
    trainer.parse_options(sys.argv)

    classes = trainer.get_class_name_from_path(trainer.options.training_path)
    training_paths = [trainer.options.training_path + '/' + c for c in classes]
    testing_paths = [trainer.options.test_path + '/' + c for c in classes]
    result_path = trainer.options.result_path
    classifier_name = trainer.options.classifier_name
    classifier_file_name = "%s.dat" % classifier_name

    # Create feature extractors and classifier
    print "Using Classifier:", classifier_name, "\n"
    extractors = [
        trainer.create_extractor('hue'),
        trainer.create_extractor('edge'),
        trainer.create_extractor('haar')
    ]
    classifier = trainer.create_classifier(classifier_name, extractors)
    trainer.set_classifier(classifier)

    # Train classifier
    print "Training classifier with sets: ", training_paths, "..\n"
    trainer.train_classifier(classes, trainer.options.training_path)
    # Test classifier
    print "Testing classifier with sets: ", testing_paths, "..\n"
    trainer.test_classifier(classes, trainer.options.test_path)

    # Visualize results of classifier by classifying every test image
    print "Visualizing test results in directory: ", result_path, "..\n"
    images = ImageSet()
    for p in testing_paths:
        images += ImageSet(p)
    #random.shuffle(images)  # shuffling images for random classification
    trainer.save_results(classifier, images, result_path)

    # Saving Classifier data
    print "Classifier data saving as: ", classifier_file_name, "..\n"
    trainer.save_classifier(classifier_file_name)

    print "Done"


if __name__ == "__main__":
    process()