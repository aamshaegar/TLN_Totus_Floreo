import spacy
from nltk import tokenize
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")
STOP_WORDS.append(',')
STOP_WORDS.append('.')
STOP_WORDS.append(';')
STOP_WORDS.append(':')
STOP_WORDS.append('-')
STOP_WORDS.append('II')
STOP_WORDS.append('di')
STOP_WORDS.append('etc')
STOP_WORDS.append("'s")
STOP_WORDS.append("Lt.")
STOP_WORDS.append('\n')
STOP_WORDS.append('\t')

nlp = spacy.load("en_core_web_sm")
CONTENT_WORD_POS = ["NOUN", "PROPN", "VERB", "ADV", "ADJ"]


def open_and_check(dataset_path):
    """
        Open a file and read all lines.
        - Throw FileNotFoundError if file not exists
        - Throw ValueError if file is void
        - param "dataset_path": path of a file
    """

    try:
        file = open(dataset_path, "r", encoding='utf8')
    except FileNotFoundError:
        print("Error! File not found...")
        exit()

    lines = file.read()
    if lines == []:
        raise ValueError("Error, void file in input!")

    file.close()
    return lines



def document_sentence_tokenize(document):
    """ 
        From a file readed as a string, tokenize this file 
        into a sequence of sentences
    """

    tokenized_document = tokenize.sent_tokenize(document)
    return tokenized_document




def sentence_parsing(sentences_tokenized):
    """ 
        This function analize all the sentences in the dataset, 
        next, it parsifies the sentence extracting for each content
        word found all the sintactic dependency.
        return a dictionary in which for each word lemma adds a list 
        of sintactic dependency
        - param "sentences_tokenized": a document tokenized per sentences
    """

    words_by_pos = dict()
    for sentence in sentences_tokenized:
        sent_parsed = nlp(sentence)
        remove_stop_words = [token for token in sent_parsed if token.text not in STOP_WORDS]

        for token in remove_stop_words:
            if token.pos_ in CONTENT_WORD_POS:
                lowerized = token.lemma_.lower()
                if not lowerized in words_by_pos:
                    words_by_pos[lowerized] = [token.dep_]
                else:
                    words_by_pos[lowerized].append(token.dep_)

    return words_by_pos



def extract_all_freatures(words_by_pos):
    """ 
        From the sentences parsified, extract all the sintactic dependency.
        Return a set of all sintactic dependency as a se of features for FCA
    """

    all_features = set()
    for el in words_by_pos:
        not_duplicates = set(words_by_pos[el])
        all_features = all_features.union(not_duplicates)

    return list(all_features)



def extract_all_concept(words_by_pos):
    """ 
        From the sentences parsified, extract all the concept as a list of 
        the content words found during the analisys of the sentences
    """

    all_concept = []
    for el in words_by_pos:
        all_concept.append(el)

    return all_concept



def make_row(concept, features_list, words_by_pos):
    """ create a single row in the matrix"""

    features = words_by_pos[concept]
    not_duplicates = list(set(features))
    position = [(features_list[i] in not_duplicates, features_list[i]) for i in range(len(features_list))]

    row = concept
    for el in position:
        row += ","
        if el[0] == True:
            row += "X"

    return row

def build_adiacent_matrix(file_path, words_by_pos):
    """
        make the adiacent matrix to perform the Formal Concept Analisys
    """

    concepts_list = extract_all_concept(words_by_pos)
    features_list = extract_all_freatures(words_by_pos)

    
    print("Built adiacent matrix in: ", file_path)
    feature = ""
    for el in features_list:
        feature += ","
        feature = feature + el


    file = open(file_path, "w", encoding="utf-8")
    file.write(feature + "\n")
    for concept in concepts_list:
        row = make_row(concept, features_list, words_by_pos)
        file.write(row + "\n")
    file.close()

    





