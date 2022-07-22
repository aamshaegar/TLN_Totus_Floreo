import time
import sys
from NER import NERtagger


__name__ = "__main__"


def main():

    if len(sys.argv) < 5:
        print("\n***************************************")
        print("Error: Bad parameter!\nUSAGE: python ner_tagger_main.py -DECODING training_set.txt test_set.txt SMOOTHING_STRATEGY")
        print("where, DECODING in [-viterbi, -baseline]")
        print("where, SMOOTHING_STRATEGY for viterbi decoding: [smoothing1, smoothing2, smoothing3] or passing a file '.txt' dataset path")
        print("where, SMOOTHING_STRATEGY for baseline decoding: a tag of the tagset, for example: ['O', 'B-MISC', 'B-ORG' ...] <br>")
        print("***************************************")
        exit()

    if sys.argv[1] == "-viterbi":

        start = time.time()
        print("\nStart reading file: ", sys.argv[2])
        ner = NERtagger(sys.argv[2])
        end = time.time()
        print("Time to read the file: ", round(end - start, 2), " seconds")
        print("Tagset: ", ner.get_tagSet())

        start = time.time()
        print("\nStarting viterbi decoding!")
        print("Smoothing strategy: ", sys.argv[4])
        ner.testing_viterbi(sys.argv[3], sys.argv[4])
        end = time.time()
        print("Time for decoding: ", round(end - start,2), " seconds")

        print()
        metrics, accuracy = ner.get_evaluating_metrics()
        print("- Accuracy: ")
        for tag in accuracy:
            print(tag, accuracy[tag])
        
        print()
        print("- Precision and recall per entity:")
        for tag in metrics:
            print(tag, metrics[tag])
        
        

    
    elif sys.argv[1] == "-baseline":

        start = time.time()
        print("\nStart reading file: ", sys.argv[2])
        ner = NERtagger(sys.argv[2])
        end = time.time()
        print("Time to read the file: ", round(end - start, 2), " seconds")
        print("Tagset: ", ner.get_tagSet())

        start = time.time()
        print("\nStarting baseline decoding!")
        print("Smoothing strategy: TAG " + sys.argv[4])
        ner.testing_baseline(sys.argv[3], sys.argv[4])
        end = time.time()
        print("Time for decoding: ", round(end - start,2), " seconds")

        print()
        metrics, accuracy = ner.get_evaluating_metrics()
        print("- Accuracy: ")
        for tag in accuracy:
            print(tag, accuracy[tag])
        
        print()
        print("- Precision and recall per entity:")
        for tag in metrics:
            print(tag, metrics[tag])

    else:
        print("Error: Bad parameter!\nUSAGE: python ner_tagger_main.py -DECODING training_set.txt test_set.txt SMOOTHING")
        exit()


if __name__ == "__main__":
    main()

