import os
import re
import numpy as np
import gensim as gs
from gensim.scripts.glove2word2vec import glove2word2vec
import scipy
import matplotlib
import matplotlib.path as path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn.preprocessing


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
        try:
            word_vector = global_vectors.wv[word_vector]
        except KeyError:
            return -999

    if np.any(std_dims != False):
        if len(std_dims) != global_vectors.wv.syn0.shape[1]:
            print("ERROR: something went wrong here std_dims, should have the same dimensions as global_vectors")
            print("length of std_dims:%f   length of glovec:%f" % (len(std_dims), global_vectors.wv.syn0.shape[1]))
        else:
            # find the last_index we keep based on the percent
            index = round(perc_dim_to_compare * len(std_dims))
            arg_important_features = np.argsort(std_dims)[0:index]
            return 1 - scipy.spatial.distance.cosine(word_vector[arg_important_features], topic_vector[arg_important_features])
    else:
        return (1 - scipy.spatial.distance.cosine(word_vector, topic_vector))


def generate_related_words(word, global_vectors, topn=10):
    """
    :param word: the word you want to generate a list of synonyms from
    :param global_vectors: gensim KeyedVectors object with wordvectors
    :return: list of related_words (default length is 10 if not put)
    """
    try:
        list_of_synonyms = global_vectors.similar_by_word(word, topn)
        related_words = []
        for synonym in list_of_synonyms:
            related_words.append(synonym[0])
        return related_words
    except KeyError:
        print(word + " is not a word in the global vector space, choose another topicword")
    else:
        raise


def analyze_related_words(related_words, global_vectors):
    """ This function prints out the cosine distance between the wordvectors and the topicvector it creates from them.
    It is meant as a tool to analyse (numerical) outliers of the words you pick to represent the group so you are sure
    that you don't portrait the wrong meaning of the topic
    """
    # look up the words in the vectorspace
    word_vectors = [global_vectors.wv[word] for word in related_words]
    # sum up the wordvectors to one big vector representing the average "meaning" of all the words
    topic_vector = sum(word_vectors)
    # Compute cosine distance from the individual wordvectors: To identify
    cosine_dist_to_topic = [(scipy.spatial.distance.cosine(topic_vector, vec)) for vec in word_vectors]
    print("Cosine distance from wordvector to topicvector:\n", dict(zip(related_words, cosine_dist_to_topic)))

    # scale the data to the range (0,1) for better vizual representation
    scaled_word_vectors = sklearn.preprocessing.minmax_scale(word_vectors, feature_range=(0.5, 1))
    # polarplot the variables
    polarplot_wordvector(wordvectors=scaled_word_vectors)


def polarplot_wordvector(wordvectors, yticks=None, ylim=None):
    # Example property
    properties = range(wordvectors[0].shape[0])
    # Choose some nice colors
    matplotlib.rc('axes', facecolor='white')
    # Make figure background the same colors as axes
    fig = plt.figure(figsize=(10, 8), facecolor='white')
    # set the radians
    t = np.arange(0, 2 * np.pi, 2 * np.pi / len(properties))
    # create colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(wordvectors)))

    for wordvector, color in zip(wordvectors, colors):
        # Use a polar axes
        axes = plt.subplot(polar=True)

        # Draw polygon representing values
        points = [(x, y) for x, y in zip(t, wordvector)]
        points.append(points[0])
        points = np.array(points)
        codes = [path.Path.MOVETO, ] + \
                [path.Path.LINETO, ] * (len(wordvector) - 1) + \
                [path.Path.CLOSEPOLY]
        _path = path.Path(points, codes)
        _patch = patches.PathPatch(_path, fill=False, color=color, linewidth=0, alpha=.1)
        axes.add_patch(_patch)
        _patch = patches.PathPatch(_path, fill=False, color=color, linewidth=2)
        axes.add_patch(_patch)
        _patch.set_edgecolor(color)

        plt.hold
    # Draw ytick labels to make sure they fit properly
    for i in range(len(properties)):
        angle_rad = i / float(len(properties)) * 2 * np.pi
        angle_deg = i / float(len(properties)) * 360
        ha = "right"
        if angle_rad < np.pi / 2 or angle_rad > 3 * np.pi / 2:
            ha = "left"
        plt.text(angle_rad, 1.04 * np.max(wordvectors[0]), properties[i], size=14, horizontalalignment=ha, verticalalignment="center")

    # Set ticks to the number of properties (in radians)
    plt.xticks(t, [])

    plt.show()

