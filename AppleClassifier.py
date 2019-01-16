#!/usr/bin/env python
from optparse import OptionParser
from SimpleCV import *


class AppleClassifier:

    def __init__(self):
        self.classifier = None
        self.options = None

    def load(self, classifier_file_name):
        self.classifier = TreeClassifier.load(classifier_file_name)

    @staticmethod
    def get_class_name_from_path(training_path):
        classes = []
        directory_list = os.listdir(training_path)
        for directory_name in directory_list:
            if os.path.isdir(training_path + '/' + directory_name):
                classes.append(directory_name)
        return classes

    def classify(self, image_file_name):
        image = Image(image_file_name)
        return self.classifier.classify(image)

    def test_classifier(self, classes):
        testing_paths = ['testing/' + c for c in classes]
        print "Test results", self.classifier.test(testing_paths, classes, verbose=False), "\n"

    def class_names(self):
        return self.classifier.mClassNames

    def parse_options(self, args):
        parser = OptionParser()
        parser.add_option("-c", "--classifier", action="store", dest="classifier_file", default="",
                          help="load classifier from file"),
        # parser.add_option("-i", "--image", action="store", dest="image_file", default="",
                          # help="classify this image file"),

        (self.options, args) = parser.parse_args(args)

        if not self.options.classifier_file or not self.options.image_file:
            parser.print_help()
            exit(0)


def main():
    apple_classifier = AppleClassifier()
    apple_classifier.parse_options(sys.argv)

    classifier_file = apple_classifier.options.classifier_file
    print "Loading classifier ..\n"
    apple_classifier.load(classifier_file)
  #  class_name = apple_classifier.classify(image_file)

    while True:
        image_name = raw_input("Enter image name or 'exit' to exit the program:\n")
        if image_name != 'exit':
            class_name = apple_classifier.classify(image_name)
            print "Apple class: ", class_name, "\n"
        else:
            break

   # print class_name


if __name__ == "__main__":
    main()
