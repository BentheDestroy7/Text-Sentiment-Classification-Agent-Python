# Benjamin Van der Ploeg
# 101139487
# COMP 3106 Asssignment 4

import os
import math 

class bag_of_words_model:
    
    def __init__(self, directory):
        # directory is the full path to a directory containing trials through state space

        # create variables
        self.number_of_documents = 0
        self.vocabulary = []
        self.array_of_arrays = []
        self.vocab_document_count = []
        self.number_of_documents_divided_by_vocab_document_count = []
        self.inverse_document_frequency_vector = []

        # loop through each .txt document in the directory
        for filename in os.listdir(directory):
            # increment the number of documents
            self.number_of_documents += 1

            # open the file by joining the filename and directory
            with open(os.path.join(directory, filename)) as file:
                # create variable for the read file
                fileRead = file.read()
                # split the read file into an array of words
                array = fileRead.split()
                # add the new array of words to an array of arrays containing the documents words
                self.array_of_arrays.append(array)
                # add words to vocabulary if it is not already there
                for string in array:
                    if string not in self.vocabulary:
                        self.vocabulary.append(string)

        # sort the vocabulary
        self.vocabulary.sort()

        # creates an array containing how many documents contain each vocab word
        # loop through vocab words
        for vocab_word in self.vocabulary:
            count = 0
            # loop through each document
            for document in self.array_of_arrays:
                checker = False
                # loop through each word in the document
                for document_word in document:
                    #check if the document word is equal to the vocab word, if yes make checker true
                    if document_word == vocab_word:
                        checker = True
                # if checker is true it means that the vocab word is in the document
                if checker:
                    count += 1
            # add the number of documents that contain the word to the array
            self.vocab_document_count.append(count)

        # divide each value in the array containg how many documents contain each vocab word by 3
        for value in self.vocab_document_count:
            self.number_of_documents_divided_by_vocab_document_count.append(self.number_of_documents / value)

        # calculate the inverse-document frequency vector
        for value in self.number_of_documents_divided_by_vocab_document_count:
            self.inverse_document_frequency_vector.append(math.log2(value))

        # Return nothing


    def tf_idf(self, document_filepath):
        # document_filepath is the full file path to a test document
        self.length_of_each_document = []
        self.count_array = []
        term_frequency_vector = []
        tf_idf_vector = []

        # open the test document and create an array of all the words in it
        with open(document_filepath) as file:
            test_document = file.read()
            test_document = test_document.split()

        # create and array that contatins how many times each vocab word is in the supplied test document
        for vocab_word in self.vocabulary:
            count = 0
            for document_word in test_document:
                if document_word == vocab_word:
                    count += 1
            self.count_array.append(count)

        # create the term frequency vector
        for count in self.count_array:
            term_frequency_vector.append(count / len(test_document))

        # create the term frequency - inverse doument frequency vector
        count = 0
        for term in term_frequency_vector:
            tf_idf_vector.append(term * self.inverse_document_frequency_vector[count])
            count += 1

        # Return the term frequency-inverse document frequency vector for the document
        return tf_idf_vector


    def predict(self, document_filepath, weights):
        # document_filepath is the full file path to a test document
        # weights is a list of weights for the artificial neuron
        self.length_of_each_document = []
        self.count_array = []
        term_frequency_vector = []
        tf_idf_vector = []
        tf_idf_vector_by_weight = []
        total_of_tf_idf_vector_by_weight = 0
        prediction = 0

        # open the test document and create an array of all the words in it
        with open(document_filepath) as file:
            test_document = file.read()
            test_document = test_document.split()

        # create and array that contatins how many times each vocab word is in the supplied test document
        for vocab_word in self.vocabulary:
            count = 0
            for document_word in test_document:
                if document_word == vocab_word:
                    count += 1
            self.count_array.append(count)

        # create the term frequency vector
        for count in self.count_array:
            term_frequency_vector.append(count / len(test_document))

        # create the term frequency - inverse document frequency vector
        count = 0
        for term in term_frequency_vector:
            tf_idf_vector.append(term * self.inverse_document_frequency_vector[count])
            count += 1

        # multiply each value in the term frequency - inverse document frequency vector by the corresponding weight
        count = 0
        for term in tf_idf_vector:
            tf_idf_vector_by_weight.append(term * weights[count])
            count += 1

        # add all of the just calculated values together
        for term in tf_idf_vector_by_weight:
            total_of_tf_idf_vector_by_weight += term

        # calculate the prediction using the sum in the logistic function
        prediction = total_of_tf_idf_vector_by_weight * -1
        prediction = math.exp(prediction)
        prediction = prediction + 1
        prediction = 1 / prediction

        # Return the prediction from the neural network model
        return prediction
