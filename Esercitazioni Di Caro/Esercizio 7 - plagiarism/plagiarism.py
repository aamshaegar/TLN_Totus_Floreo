# metodi per costruire i vettori TF-IDF
# metodi per longest common subsequenze?
# metodi per misurare il plagio

import ast
from dataPreprocessing import *
from numpy import vectorize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def open_and_check(file_python):
    """
        Open a python file
        - Throw FileNotFoundError if file not exists
    """

    try:
        file = open(file_python, "r", encoding='utf8')
    except FileNotFoundError:
        print("Error! File not found...")
        exit()

    return file




def open_and_read(file_python):
    """ Read the python file as a string"""

    file = open_and_check(file_python)
    lines = file.read()
    if lines == []:
        raise ValueError("Error, void file in input!")

    file.close()
    return lines



def open_and_readlines(file_python):
    """ Read the python file as a list of code line"""

    file = open_and_check(file_python)
    lines = file.readlines()
    if lines == []:
        raise ValueError("Error, void file in input!")

    file.close()

    new = [line.replace("\n", "") for line in lines]
    return new




def select_all_relevants_words(python_string_code):
    """
        Parsing a python code with an Abstract Syntax Tree parser.
        The python_string_code is a python code readed as a string.
        The function return a list of all the relevant words found in the 
        python input code.

        A word is a relevant word if it is a:
        - module name
        - variable name
        - class name
        - method or function name
        - exception raised
    """

    root = ast.parse(python_string_code)
    names = [node.id for node in ast.walk(root) if isinstance(node, ast.Name)]
    return names



def select_all_comment(file_python):
    """ Get all the comment """

    python_code_lines = open_and_readlines(file_python)
    comments = []
    for line in python_code_lines:
        if "#" in line:
            line_splitted = line.split("#")
            bags = bag_of_words_mapping(line_splitted[1])
            for el in bags:                
                comments.append(el)
    
    return comments



def TF_IDF_trasmorm(python_code1, python_code2):
    """
        Perform a TF-IDF vectorize rappresentation of the source 
        codes passed as input. Return the cosine similarity between
        the two vectors to understand if there is a plagiarism attempt

        - param python_code1: the relevant terms rappresentation for the first file
        - param python_code2: the relevant terms rappresentation for the soucpicious file
    """

    join1 = " ".join(python_code1)
    join2 = " ".join(python_code2)

    vectors = TfidfVectorizer().fit_transform([join1, join2]).toarray()
    similarity_score = cosine_similarity([vectors[0], vectors[1]])[0][1]

    return similarity_score




def longest_common_subsequence(origin, souspicious):
    """
        Calculate the longest common subsequence between two sequences of terms.
        The algorithm find the length of longest subsequence present in 
        both of them. A subsequence is a sequence that appears in the same 
        relative order, but not necessarily contiguous. 

        - param origin: origin python code readed as a string
        - param souspicious: soucpicious python code readed as a string
    """
    
    len_origin = len(origin)
    len_sousp = len(souspicious)

    # define a matrix with len_origin rows and len_sousp coulums
    # initialize the matrix with 0
    # we have interest to store in that matrix only the number of the match, not the subsequence
    matrix = [[0 for x in range(len_sousp)] for x in range(len_origin)]
    
    """
        we check for all character of the origin code if it match with 
        one of the character of the souspicious code.
        Every time we see that the character in position i in origin is equal to the 
        character in position j for the souspicious string, we increment matrix[i][j]
        For every other confront we report in matrix[i][j] the maximum LCS (max(matrix[i-1][j], matrix[i][j-1]))

    """

    for i in range(len_origin):
        for j in range(len_sousp):
            
            if origin[i] == souspicious[j]: 
                
                # to control the first match. Every other match are in the else case
                if i == 0 or j == 0:
                    matrix[i][j] = 1
                else:
                    matrix[i][j] = matrix[i-1][j-1] + 1
            else:
                matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1])

    
    # the longhest common subsequence number is stored in the last cell of the matrix
    lcs = matrix[-1][-1]
    score = lcs

    return score
