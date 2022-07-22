from segmentation import *
import sys

__name__ = "__main__"

def main():

    
    if len(sys.argv) < 3:
        print("\n***************************************")
        print("Error: Bad parameter!\nUSAGE: python main.py CORPUS BLOCK_SIZE")
        print("where, CORPUS is the path of the corpus which contains some sentences")
        print("where, BLOCK_SIZE is the size of every block of sentences at the initial stage")
        print("***************************************\n")
        exit()
    
    path = sys.argv[1]
    block_size = int(sys.argv[2])

    document = open_and_check(path)
    document_sentences = document_sentence_tokenization(document, block_size)

    similarity_list = []
    for i in range(len(document_sentences)-1):

        # retrieve near blocks
        block_i = document_sentences[i]
        block_next_i = document_sentences[i+1]

        # calculate bag of words
        bags_for_i = block_preprocessing(block_i)
        bags_for_next_i = block_preprocessing(block_next_i)

        # return lexical overlap similarity
        similarity = inter_block_similarity(bags_for_i, bags_for_next_i)
        similarity_list.append((similarity,i))


    # retrieve list of local minimum score
    minumum_locals = []
    for i in range(0,len(similarity_list)):
        
        # first
        if (i == 0):
            if similarity_list[i][0] < similarity_list[i+1][0]:
                minumum_locals.append(similarity_list[i])
        # end
        elif(i == len(similarity_list)-1):
            if similarity_list[i - 1][0] >= similarity_list[i][0]:
                minumum_locals.append(similarity_list[i]) 
        # other
        else: 
            if similarity_list[i-1][0] >= similarity_list[i][0] and similarity_list[i+1][0] > similarity_list[i][0]:
                minumum_locals.append(similarity_list[i])

    minumum_locals = [el[1] for el in minumum_locals]
    if not (len(similarity_list)-1) in minumum_locals:
        minumum_locals.append(len(similarity_list)-1)

    new_document_sentences = []
    init = 0
    end = 0
    
    # redefine all group of sentences
    for i in range(len(minumum_locals)):

        end = minumum_locals[i]
        sentences = document_sentences[init:end]
        document = []
        for el in sentences:
            for sent in el:
                document.append(sent)
        new_document_sentences.append(document) 
        init = end

    # if there are other sentences to consider in the list
    if len(document_sentences[end:]) > 0:
        doc = []
        sentences = document_sentences[end:]
        for el in sentences:
            for sent in el:
                doc.append(sent)
        new_document_sentences.append(doc)


    write_result(document_sentences,new_document_sentences)
    print("\nDocument segmented \n")
    for sent in new_document_sentences:
        print(sent)
        print()



if __name__ == "__main__":
    main()