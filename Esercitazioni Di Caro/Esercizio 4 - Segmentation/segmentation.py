from DataPreprocessing import *
from nltk import tokenize


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



def document_sentence_tokenization(document, block_size):
    """ Tokenize all paragrah into a list of sentence of "block_size" size """

    document_sentences = []
    tokenized_document = tokenize.sent_tokenize(document)
    size = int(len(tokenized_document) / block_size)


    init = 0
    end = 0
    for i in range(size):
        init = (i * block_size)
        end = init + block_size
        if end < len(tokenized_document):
            document_sentences.append(tokenized_document[init:end])

    if end < len(tokenized_document):
        document_sentences.append(tokenized_document[end:len(tokenized_document)])
    
    return document_sentences
    


def block_preprocessing(sentences_block):
    """ Execute a bag of words of a block of sentences """

    bags = set()
    for sent in sentences_block:
        bags = bags.union(bag_of_words_mapping(sent))

    return bags



def inter_block_similarity(sentence1, sentence2):
    """
        Calculate the similarity between two bag of words sets from the readed data.
        - param "sentence1": a list of terms
        - param "sentence2": a list of terms
    """
    
    # calculate the overlaps
    lexical_overlap = sentence1.intersection(sentence2)
    number_overlap = len(lexical_overlap)

    # Normalize the denominator with the length of the minor sentence
    min_length = 0
    len_sent1 = len(sentence1)
    len_sent2 = len(sentence2)

    if len_sent1 > 0 and len_sent2 > 0:
        min_length = min(len_sent1, len_sent2)         
                                          
    return number_overlap / min_length  



def write_result(document_sentences,new_document_sentences):
    """ Print all the result in 2 separated files """

    with open('./output/segmented_document.txt', "w", encoding="utf8") as f:
        for paragraph in new_document_sentences:
            for s in paragraph:
                f.write(s)
                f.write("\n")
            f.write('\n' * 3)

    with open('./output/initial_segments.txt', "w", encoding="utf8") as f:
        for paragraph in document_sentences:
            for s in paragraph:
                f.write(s)
                f.write("\n")
            f.write('\n' * 3)
