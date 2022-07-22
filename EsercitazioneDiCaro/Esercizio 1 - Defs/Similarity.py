from DataPreprocessing import *



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

    lines = file.readlines()
    if lines == []:
        raise ValueError("Error, void file in input!")

    file.close()
    return lines



def read_csv_file(csv_path):
    """
        read from a comma separated file all the data used for calculate similarities
        param "csv_path": path of a csv file
    """

    lines = open_and_check(csv_path)
    terms = dict()

    for line in lines:
        sentence_splitted = line.split(",")
        if not sentence_splitted[0] in terms:
            terms[sentence_splitted[0]] = sentence_splitted[1:]
                
    return terms



def preprocessing(terms):
    """
        For all the definition of a single term, 
        this function creates a bag of words replacing
        the list of term by a set of relevant words.
        - param "terms": dictionary of associations term - definitions
    """
    for term in terms:
        for i in range(0,len(terms[term])):
            sentence = bag_of_words_mapping(terms[term][i])
            terms[term][i] = sentence



def bag_of_words_similarity(sentence1, sentence2):
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
