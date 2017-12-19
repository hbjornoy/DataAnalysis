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
import pickle
import pandas as pd


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
            return np.nan

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
    """
    This plotting function is only used as a analysistool while working out how to define similarity to topic. It is not a integral part of our code. But it shows how we work with vizualizing to understand the problemspace.
    """
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


def vocabulary_calculate_topics(words_defining_topics, name_of_topics, global_vectors, keep_const=0.45,full_word_list_path="full_word_list.pkl"):
    """
    calculate each words similarity to a topic
    """
    # loop through the topics, create the topic, and group same topics in the same tuple
    topic_vectors = []
    for topic in words_defining_topics:
        topic_vectors.append(create_topic(topic, global_vectors))
    
    full_word_list = pickle.load( open( full_word_list_path, "rb" ) )
    vocab_topics = full_word_list.copy(deep=True)
    perc_dim_to_compare = 0.4

    for topics, topic_name in zip(topic_vectors, name_of_topics):

        print(topic_name)

        topic_scores = []
        for word in vocab_topics.Word:
            topic_scores.append(calculate_topic_similarity(word, topics[0],
                                global_vectors, std_dims=topics[1], perc_dim_to_compare=perc_dim_to_compare))

        # giving a suitable cloumnname
        columnname = "topic_" + topic_name
        topic_column = pd.Series(topic_scores, index=full_word_list.index)
        topic_column = topic_column.apply(lambda x: 1 if x>0.45 else 0)
        vocab_topics[columnname] = topic_column
        
    vocab_topics = vocab_topics.dropna()
    
    return vocab_topics
    
def score_song(words_freq, vocab_topics, column):
    """
    Calculates the score per song given the unique words in the song and their frequency, 
    it must also know the vocabulary of the topic
    """
    occurence_words_not_in_vocab = 0
    total_words_in_song = 0
    nr_words_per_song = 0
    column_score = 0
    words_freq = list(words_freq)
    for word_info in words_freq:
        [word_index, freq] = word_info.split(':')
        total_words_in_song += int(freq)
        try:
            word_score = vocab_topics.get_value(int(word_index),column)
            column_score += word_score*int(freq)
            nr_words_per_song += int(freq)
        except KeyError:
            occurence_words_not_in_vocab += int(freq)
    if nr_words_per_song is not 0:
        column_score = column_score/nr_words_per_song
    else:
        column_score = 'NaN'
    return column_score, occurence_words_not_in_vocab, total_words_in_song


def score_songs(matches, vocab_topics, column_startwith='topic'):
    """
    Calculates the score for every song and add infomation about the topics to the song dataframe
    """
    
    #Fetch the columnnames of the topics
    column_names = [col for col in vocab_topics.columns if col.startswith(column_startwith)]

    #Applying the polarity analysis for the bags of words
    for column in column_names:
        occ_sum = 0
        tot_sum = 0
        for i in matches.index: 
            if np.all(score_song(matches.at[i, 'words_freq'], vocab_topics, column) != None):
                matches.at[i, column], occ, tot = score_song(matches.at[i, 'words_freq'], vocab_topics, column)
                occ_sum += occ
                tot_sum += tot
    
    return matches, occ_sum, tot_sum 
