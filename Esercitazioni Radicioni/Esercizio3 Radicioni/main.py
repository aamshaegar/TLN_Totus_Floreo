import sys
import os
from pathlib import Path
from summarization import *

__name__ = "__main__"


def main():


    if len(sys.argv) < 4:
        print("\n***************************************")
        print("Error: Bad using of parameter!\nUSAGE: python main.py FILE PERC METHOD")
        print("where, FILE = path of the file to summarization")
        print("where, PERC = compression rate (select one of 10, 20, 30)")
        print("where, METHOD = method for the definition of the topic [title, cue]")
        print("***************************************\n")
        exit()


    # check parameter
    file_path = sys.argv[1]
    percentage = int(sys.argv[2])
    topic = sys.argv[3]

    if (percentage not in [10,20,30]):
        print("\n***************************************")
        print(" Error: Bad using of PERC parameter!\n USAGE: select one of (10, 20, 30)")
        print("***************************************\n")
        exit()

    if(not topic in ["title", "cue"]):
        print("\n***************************************")
        print(" Error: Bad using of METHOD parameter!\n USAGE: select one of [title, cue]")
        print("***************************************\n")
        exit()


    # retrieve the NASARI vectors list
    nasari_dict = util_nasari()
    document = parse_document(file_path)
    summarization = automatic_summarization(document, nasari_dict, percentage, topic)

    # Evaluate the automatic summarization performance
    BLUE = BLUE_evaluation(document, summarization, percentage)
    ROUGE = ROUGE_evaluation(document, summarization, percentage)


    # check if file exists
    file_name = Path(file_path).stem
    output_file_path = "./output/" + str(percentage) + '_' + file_name + "_" + topic + "Topic" + ".txt"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)


    # write output summarization file
    output = open(output_file_path, 'a', encoding='utf-8') 
    output.write("------------------------------------------------------------------------" + "\n")
    output.write("bilingual evaluation understudy (BLEU):  " + str(BLUE) + "%" + "\n")
    output.write("Recall-Oriented Understudy for Gisting Evaluation (ROUGE):  " + str(ROUGE) + "%" + "\n")
    output.write("------------------------------------------------------------------------" + "\n" + "\n")
    
    for paragraph in summarization:
        output.write(paragraph)
        output.write("\n")
    
    output.close()

if __name__ == "__main__":
    main()