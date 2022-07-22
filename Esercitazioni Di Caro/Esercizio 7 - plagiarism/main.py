from plagiarism import *
import sys


__name__ = "__main__"



def main():

    
    if len(sys.argv) < 3:
        print("\n***************************************")
        print("Error: Bad using of parameter!\nUSAGE: python main.py ORIGIN  SOUSPICIOUS ")
        print("where, ORIGIN = path of the python original file ")
        print("where, SOUSPICIOUS = path of the python file which is supposed to be a plagiarism")
        print("***************************************\n")
        exit()

    
    # path of the file
    origin_file = sys.argv[1]
    souspicious_file = sys.argv[2]

    # read and check the file
    origin_code_string = open_and_read(origin_file)
    souspicious_code_string = open_and_read(souspicious_file)

    # select all relevant terms
    relevant_terms1 = select_all_relevants_words(origin_code_string)
    relevant_terms2 = select_all_relevants_words(souspicious_code_string)

    # select all the content words in the comments
    origin_comments = select_all_comment(origin_file)
    souspicious_comments = select_all_comment(souspicious_file)


    # make a context for the TDF-IDF vectors with all the relevant
    # terms and the comments
    relevant_terms1 += origin_comments
    relevant_terms2 += souspicious_comments


    # calculate the cosine similarity between the two TDF-IDF vector
    # defined on the context of the two codes
    similarity = TF_IDF_trasmorm(relevant_terms1, relevant_terms2)
    
    # calculate the longest common subsequence between the two code files.
    tokenized_origin = tokenize_sentence(origin_code_string)
    tokenized_souspicious = tokenize_sentence(souspicious_code_string)
    lcs_score = longest_common_subsequence(tokenized_origin, tokenized_souspicious)
    # we normalize the lcs_score to the length of the souspicious code in order to
    # obtain the proportion of the length of the longest common line of code in the documents
    
    rate_lcs = lcs_score / len(tokenized_souspicious)
    print("\nPlagiarism TDF-IDF: ", 100 * round(similarity,3), "%")
    print("Plagiarism Longest common subsequence: ",100 * round(rate_lcs,3), "%\n")

    if similarity > 0.5 and rate_lcs > 0.5:
        print("We have a plagiarism attempt!")
    elif similarity > 0.5 or rate_lcs > 0.5:
        print("The two programs seems to be a bit similar!")
    elif similarity > 0.2 or rate_lcs > 0.2:
        print("The two programs seems to be not much similar")
    else:
        print("We have no plagiarism attempt!")
    print()

        



if __name__ == "__main__":
    main()