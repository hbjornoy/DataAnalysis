import os
import re
import numpy as np
import gensim as gs
from gensim.scripts.glove2word2vec import glove2word2vec
import scipy


def create_gensim_wv_from_glove(path_to_glove_folder):
    """
    :param path_to_glove_folder: Go to Stanfords website https://nlp.stanford.edu/projects/glove/ and download their twitterdataset,
            put it in the same folder as this function and write the path to it as the input of this function
    :return: nothing but creates a .txt-file with the gensim object, afterwards you can delete the original glove-files
            and keep the created gensim___.txt files and use the function load_gensim_global_vectors(path_to_global_vectors) to load them in.
    """

    for filename in os.listdir(path_to_glove_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(path_to_glove_folder, filename)
            dim = get_dim_of_file(filename)
            name_of_filecreated = "gensim_global_vectors_"+ dim + "dim.txt"

            # spits out a .txt-file with the vectors in gensim format
            glove2word2vec(glove_input_file=filepath, word2vec_output_file="gensim_glove_"+ dim + "dim.txt")
            print(name_of_filecreated)
            continue
        else:
            continue

def get_dim_of_file(filename):
    removed_27 = filename.replace("27", "")
    # after removing 27 it should only be dim_nr. that is numerical, create regex that filter nonnumericals out
    non_decimal = re.compile(r'[^\d]+')
    dim = non_decimal.sub('', removed_27)
    return dim

def load_gensim_global_vectors(path_to_global_vectors):
    # load the gensim object
    glove_model = gs.models.KeyedVectors.load_word2vec_format(path_to_global_vectors, binary=False)

    # delete unneccessary parts to save RAM in computations
    global_vectors = glove_model.wv
    del glove_model

    return global_vectors

def create_topic(words_defining_topic, global_vectors):
    # look up the words in the vectorspace
    word_vectors = np.array([global_vectors.wv[word] for word in words_defining_topic])
    # sum up the wordvectors to one big vector representing the average "meaning" of all the words
    topic_vector = sum(word_vectors)

    std_dims = [np.std(word_vectors[:, i]) for i in range(word_vectors.shape[1])]

    return topic_vector, std_dims


def calculate_topic_similarity(word_vector, topic_vector, global_vectors, std_dims=False, perc_dim_to_compare=0.5):
    """

    :param word_vector: the word or wordvector you want to calculate topic_score for
    :param std_dims: a list with standard deviation in the different dimensions for the topic
    :param perc_dim_to_compare: How many percent of the dimensions with the least variance do you want to calculate cosine similarity on
    :return: a score between 0-1 of how correlated the word is with the topic racism

    """
    if type(word_vector) is str:
        word_vector = global_vectors.wv[word_vector]

    if std_dims != False:
        if len(std_dims) != global_vectors.wv.syn0.shape[1]:
            print("ERROR: something went wrong here std_dims, should have the same dimensions as global_vectors")
        else:
            # find the last_index we keep based on the percent
            index = round(perc_dim_to_compare * len(std_dims))
            print("index: ", index)
            arg_important_features = np.argsort(std_dims)[0:index]
            print("args", arg_important_features)
            return 1 - scipy.spatial.distance.cosine(word_vector[arg_important_features], topic_vector[arg_important_features])
    else:
        return (1 - scipy.spatial.distance.cosine(word_vector, topic_vector))

def generate_related_words(word, global_vectors, topn=10):
    """
    :param word: the word you want to generate a list of synonyms from
    :param global_vectors: gensim KeyedVectors object with wordvectors
    :return: list of related_words (default length is 10 if not put)
    """

    list_of_synonyms = global_vectors.similar_by_word(word, topn)
    related_words = []
    for synonym in list_of_synonyms:
        related_words.append(synonym[0])
    return related_words





if __name__ == '__main__':
