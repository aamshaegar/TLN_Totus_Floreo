import re


def remove_punctuation(sentences):
    """ Remove punctuation and multiple spaces from a string """
    
    new_sentences = []
    for sent in sentences:
        new_sentences.append(re.sub('\s\s+', ' ', re.sub(r'[^\w\s]', '', sent)))
    return new_sentences



def remove_new_line(sentences):
    """ Remove all \n symbols"""
    return [sent.replace("\n", "") for sent in sentences if sent.replace("\n", "") != ""]



def to_lower(sentences):
    """ lowerize all sentences"""
    return [sent.lower() for sent in sentences]



def remove_tag_s(sentences):
    """ Revove all "<s></s>" tags from the sentences of the corpus"""
    
    new_sentences = []
    for sent in sentences :
        if len(sent.split('<s>')) > 1:
            new_sentences.append(sent.split('<s>')[1].replace('</s>','').strip())
        elif len(sent.split('<p>')) > 1:
            new_sentences.append(sent.split('<p>')[1].replace('</p>','').strip())
    return new_sentences



def open_and_check(dataset_path):
    """ 
        open a file and check its consistency. 
        We accept only a sketch engine corpora.
        Throw FileNotFoundError if dataset_pat does not exist
        Throw ValueError if the dataset is void it is not a sketch engine dataset.
    """

    try:
        file = open(dataset_path, "r", encoding='utf8')
    except FileNotFoundError:
        print("Error! File not found...")
        exit()

    lines = file.readlines()
    file.close()

    if lines == []:
        raise ValueError("Error, void file in input!")


    check = False
    for line in lines:
        if len(line.split('<s>')) > 1 or len(line.split('<p>')) > 1:
            check = True
            break

    if not check:
        raise ValueError("Error, not sketch engine corpus!")

    return lines



def read_file(file_path):
    """ 
        Read all the sentences in the file_path.
        Execute some preprocessing on the read sentences
        Apply "remove_punctuation", "remove_new_line", "to_lower" and remove_tag_s preprocessings
    """

    sentences = open_and_check(file_path)
    sentences = remove_tag_s(sentences)
    sentences = remove_new_line(sentences)
    sentences = remove_punctuation(sentences)
    sentences = to_lower(sentences)
    
    return sentences
