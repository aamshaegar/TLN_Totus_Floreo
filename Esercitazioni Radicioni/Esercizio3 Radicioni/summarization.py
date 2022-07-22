import math
from collections import Counter
from preprocessing import *


#-----------------------------------------------
############ UTILITY FUNCTIONS #################
#-----------------------------------------------



def open_bonus_stigma_words():
    """
       this function read from two files all the bonus words and stigma words.
       Bonus and Stigma words are used for scoring the paragraphs of the document,
       to find all the relevant terms to make a topic. 
    """

    bonus_word = []
    stigma_word = []
    
    file = open("./util/bonus_words.txt", "r", encoding="utf-8")
    for line in file.readlines():
        row = line.replace("\n", "")
        bonus_word.append(row)
    file.close()

    file = open("./util/stigma_words.txt", "r", encoding="utf-8")
    for line in file.readlines():
        row = line.replace("\n", "")
        stigma_word.append(row)
    file.close()

    return bonus_word, stigma_word



def parse_document(file_path):
    """
    Check if the file exists, read the file as a list of paragraph and remove all the blank lines
    :param file: input document
    :return: a list of all document's paragraph (list of string).
    """

    try:
        file = open(file_path, "r", encoding='utf8')
    except FileNotFoundError:
        print("Error! File not found...")
        exit()

    data = file.read()
    file.close()

    # check if the file is not empty
    if data == []:
        raise ValueError("Error, void file in input!")

    document = []
    lines = data.split('\n')

    for line in lines:
        line = line.replace("\n", "")  # deletes the final "\n" character.
        if line != '' and '#' not in line:
            document.append(line)

    return document



def util_nasari():
    """
    It takes the Nasari input file, and it converts into a Python dictionary.
    :return: a dictionary representing the Nasari input file. Fomat: {word: {term:score}}
    """

    nasari_dict = {}
    file = open("./util/dd-small-nasari-15.txt", "r", encoding='utf8')
    
    for line in file.readlines():
        line_split = line.replace("\n", "")
        splits = line_split.split(";")
        vector_dict = {}

        for term in splits[2:]:
            k = term.split("_")
            if len(k) > 1:
                vector_dict[k[0]] = k[1]

        nasari_dict[splits[1].lower()] = vector_dict

    file.close()
    return nasari_dict



def create_context(nasari_vec, text):

    """
    It creates a list of a list of Nasari's vectors based on the document's most 
    relevant terms. The function exectutes a bag of word of the text passed 
    as parameter. Every word in this bag is linked with one or more NASARI vectors. 
    :param text: the list of text's terms
    :param nasari: Nasari dictionary
    :return: list of a list of Nasari's vectors.
    """

    tokens = bag_of_words_mapping(text)
    tokens_vectors = []

    # finding all the NASARI vectors that are associated to a single term
    for word in tokens:
        word_vectors = [nasari_vec[term] for term in nasari_vec if term == word]
        tokens_vectors.append(word_vectors)

    return tokens_vectors



def title_topic(document):
    """
    A simple criteria to find all the relevant terms in the document.
    :param document: input document
    :return: the paragraph in which there are all the relevant terms.
    """

    return document[0]



def cue_phrases_topic(document):
    """ 
        Cue phrases criteria to find the most relevant term in the document.
        most of the relevant terms are in the paragraph with the higher
        score in term of more bonus word and less stigma words counted
    """

    scores_list = []
    bonus_words, stigma_words = open_bonus_stigma_words()

    for i in range(len(document)): 
        
        # preprocess a paragraph without remove duplicates
        paragraph = document[i]
        punctuation = remove_punctuation(paragraph)
        tokens = lemmatization(punctuation)

        # calculate local score counting for every bonus words match +1 and for every stigma words match -1    
        local_score = 0
        for token in tokens:
            if token in bonus_words: 
                local_score += 1
                #print("BONUS: ", token, local_score)
            elif token in stigma_words: 
                local_score -= 1
                #print("STIGMA: ", token, local_score)

        scores_list.append((local_score, paragraph))

    # sorts paragraph for high score. 
    # make the NASARI vector from the bag of words of the higher scored paragraph
    sorted_paragraph = sorted(scores_list, key=lambda x: x[0], reverse=True)
    relevant_paragraph = sorted_paragraph[0][1]
    
    return relevant_paragraph
    


def select_topic(document, topic):
    """ 
        Select the most relevant terms in the document by using two relevant criteria: 
        1) title method: most of the relevant terms are in the title of the document, i.e in the first line of the document.
        2) cue phrases: most of the relevant terms are in the paragraph that are high score in term of more bonus word and less stigma words
    """

    # relevant criteria
    if topic == "title":
        return title_topic(document)
    elif topic == "cue":
        return cue_phrases_topic(document)
    else:
        print("no valid topic method!")
        exit()




#-------------------------------------------------------------
############ Conceptual similarity functions #################
#-------------------------------------------------------------


def rank(feature, nasari_vectors):
    """ return the rank of a feature of a NASARI vector """

    i = 1
    for word in nasari_vectors:
        if word == feature:
            return i
        i += 1
            

def weighted_overlap(ctx_vector, top_vector):
    """
        Conceptual similarity metric for NASARI vectors.
        The similarity is computed considering the features in common
        betweeen the two vector in input
        :param ctx_vector: Nasari vector
        :param top_vector: Nasari dictionary
        :return: weighted overlap similarity
    """

    # calculate the overlap between the features
    topic_features = set(top_vector.keys())
    paragraph_features = set(ctx_vector.keys())
    common_features = list(topic_features.intersection(paragraph_features))

    overlap = 0
    if len(common_features) > 0:

        numerator = 0
        for feature in common_features:
            numerator += (1 / (rank(feature, ctx_vector) + rank(feature, top_vector)))

        denominator = 0
        for i in range(len(common_features)):
            denominator += 1 / (2 * (i+1))
        
        overlap = numerator / denominator
    
    return overlap



def max_similarity(context_list, topic_list):
    """ 
        Calculate max similarity betweeen all the NASARI vectors of the two terms
        :param context_list: List of list of Nasari vector
        :param topic_list: List of list of Nasari vector
        :return: max similarity
    """

    overlaps = [0]
    for topic_vect in topic_list:
        for context_vect in context_list:
            overlaps.append(math.sqrt(weighted_overlap(context_vect, topic_vect)))
        
    return max(overlaps)




#-----------------------------------------------------
############## Automatic summarization ###############
#-----------------------------------------------------


def automatic_summarization(document, nasari_dict, percentage, topic):
    """
    Applies summarization to the given document, with the given percentage.
    The resulting summarized document maintain the same paragraphs position with 
    the original one
    :param document: the input document
    :param nasari_dict: Nasari dictionary
    :param percentage: reduction percentage
    :return: document summarized.
    """

    # list of the NASARI vectors taken by the relevant term in the title
    topic = select_topic(document, topic)
    nasari_topic = create_context(nasari_dict, topic)
    paragraphs = []
    
    # for each paragraph, except the title (document[0])
    for i in range(1, len(document)):

        # list of NASARI vector from the content word of the paragrah
        paragraph = document[i]
        context = create_context(nasari_dict, paragraph)
        total_overlap = 0
        overlaps_count = 0
    
        # Computing WO confronting each content word inside the paragraph with 
        #   the content word in the topic
        for ctx_vector in context:
            for top_vector in nasari_topic:
                total_overlap = total_overlap + max_similarity(ctx_vector, top_vector)
                overlaps_count += 1

        # assign for each paragraph a score as a mean of the total overlap for
        # the paragraph and the total number of overlap
        if overlaps_count > 0:
            average_paragraph_overlap = total_overlap / overlaps_count
            paragraphs.append((average_paragraph_overlap, paragraph, i))


    # number of paragraphs to mantain after reduction
    paragraphs_num = len(paragraphs) - int(round((percentage / 100) * len(paragraphs), 0))

    # we mantain only the first 'paragraphs_num' paragraphs in order of importance (i.e. average overlap)
    sorted_paragraphs = sorted(paragraphs, key=lambda x: x[0], reverse=True)
    reduced_paragraphs = sorted_paragraphs[:paragraphs_num]
    
    # Ordering paragraph maintaining the original position. remove scores associated
    positioned_paragraph = sorted(reduced_paragraphs, key=lambda x: x[2])
    reduced_paragraphs = [par[1] for par in positioned_paragraph] 

    # append title to the paragraph list
    summarization = []
    summarization.append(document[0]) 
    for paragraph in reduced_paragraphs:
        summarization.append(paragraph) 

    return summarization




#-----------------------------------------------------
################ Evaluation Metrics #################
#-----------------------------------------------------



def get_relevant_terms(document, percentage):
    """
        Get all relevant terms in the paragrahs that shared some of the relevant 
        word in the introduction and at the end of the document.
        :param document: the input document
        :param percentage: reduction percentage
        :return: set of the relevant terms in the document.
    """

    # select the topic
    sentences = document[0]
    sentences += document[len(document)-1]
    topic = bag_of_words_mapping(sentences)
    
    # term frequency list
    relevant_terms = []

    # select all the relevant paragraphs as paragraphs that cointain some of
    # the relevant term in the topic
    for i in range(1, len(document)):

        line = document[i]
        new_line = bag_of_words_mapping(line)

        found = False
        for term in topic:
            if term in new_line and not found:
                relevant_terms += new_line
                found = True
                             
    # remove duplicate
    return set(relevant_terms)



def BLUE_evaluation(document, summary_document, percentage):
    """
        Perform the BLUE evalution metric considering the bag of words 
        of the original document and its summarization. 
        BLUE is a precision metric. It is defined as:
        |{relevant_document} & {retrieved_document}| / |{retrieved_document}|

        :param document: the input document
        :param summary_document: summarized document
        :param percentage: reduction percentage
        :return: a percentuage of precision metric.
    """

    # take the relevant terms from the original document and the candidate
    #   terms in the summarized document
    relevant_document = get_relevant_terms(document, percentage)
    retrieved_document = bag_of_word_for_files(summary_document)

    numerator = len(list(relevant_document.intersection(retrieved_document)))
    denominator = len(list(retrieved_document))

    if denominator > 0: return 100 * round(numerator / denominator, 2)
    else: return 0




def ROUGE_evaluation(document, summary_document, percentage):
    """
        Perform the ROUGE evalution metric considering the bag of words 
        of the original document and its summarization. 
        ROUGE is a recall metric. It is defined as:
        |{relevant_document} & {retrieved_document}| / |{relevant_document}|

        :param document: the input document
        :param summary_document: summarized document
        :param percentage: reduction percentage
        :return: a percentuage of recall metric.
    """

    relevant_document = get_relevant_terms(document, percentage)
    retrieved_document = bag_of_word_for_files(summary_document)

    numerator = len(list(relevant_document.intersection(retrieved_document)))
    denominator = len(list(relevant_document))

    if denominator > 0: return 100 * round(numerator / denominator, 2)
    else: return 0