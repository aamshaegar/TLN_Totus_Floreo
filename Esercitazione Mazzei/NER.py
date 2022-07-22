import numpy
import math
import re

class NERtagger:
    """
        A class used to perform Named Entity Reconition 

        Public methods:
            get_tagSet(): return the tagset readed during the learning process
            get_sentence_number(): return the number of all sentences readed during the learing process
            get_evaluating_metrics(): Return 2 values, a dictionary with all metrics used to evaluating the Tagger, sorting per entity, and the accuracy of the tagger
        
        On __init__ (datasetPath):
            Performs the learning on a training dataset (datasetPath) organized in CoNLL style 
            USAGE DATASET: ID, WORD, TAG
    """
    
    def __init__(self, dataset_path):
        """
            Args:
                [Optionally] dataset_path(str): the path of a training dataset uset to perform the learning 
        """
        self.__labelled_dataset = []
        self.__tag_set = []
        self.__tag_entity = dict()
        self.__tag_occurrences = dict()            
        self.__tag_initial_occurrences = dict()     
        self.__tag_final_occurrences = dict() 
        self.__start_probability = dict()
        self.__final_probability = dict()
        self.__transitions = dict()
        self.__emissions = dict()
        self.__baseline_tag = dict()
        self.__statistics = {}
        self.__sentence_number = 0
        self.__transition_smoothing = 0.00000000000000000001

        self.__metrics = {"ORG": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "PER": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "MISC": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "LOC": {"COR":0,  "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}}
        
        self.__accuracy = { "GENERAL": {"ACCURACY":0},
                            "ORG": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "PER": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "MISC": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "LOC": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "O": {"TOT": 0, "COR": 0 , "ACCURACY":0}}
        self.__last_smoothing = ""
        self.__learning(dataset_path)

        
    def get_tagSet(self):
        """
            Return the tag set reading from the training set
        """
        return self.__tag_set


    def get_sentence_number(self):
        """
            Return the number of the sentences of the training set
        """
        return self.__sentence_number


    def get_evaluating_metrics(self):
        """
            Return 2 values: a dictionary with all metrics used to evaluating the Tagger sorting per entity and the accuracy of the tagger
            For each entity, the dictionary specifies 7 metrics:
                 COR: Number of correct entity tagged for all sentences in the test_set
                 MIS: Number of entity that are missed by the tagger
                 HYP: Number of entity hypothesized by the tagger
                 PRECISION: Precision metric computing as: COR / (COR + HYP)
                 RECALL: Recall metric computing as: COR / (COR + MIS)
                 F1SCORE: F1-score is the harmonic mean of precision and recall
            
            USAGE: call this method after testing NER on a test set
        """
        return self.__metrics, self.__accuracy


    def viterbi(self, observations, smoothing):
        """
            Viterbi algorithm for decoding.
            
            Args:
                observations(list str): a list of a observations
                smoothing(str): the declaration of a smoothing strategy for calculate unknown words emission probabilities. 
                Accepted values: ['smoothing1', 'smoothing2', 'smoothing3'] or passing a file '.txt' dataset path"
               
            Exceptions:
                ValueError exception: if dataset is incorrect formatting as CoNLL style
                ValueError exception: for incorrect smoothing parameter
                FileNotFound exception: in dataset not exists

            Returns:
                best_path(list str): a list of a tag used to tagging the observation passed on args
        """
  
        # for multiple execution on the same NERtagger istance
        if self.__last_smoothing != smoothing:
            self.__last_smoothing = smoothing
            self.__emissions = dict()
        try:
            self.__all_emission_probability(observations, smoothing)
        except ValueError as e:
            print(e)
            exit()
          
        #Initialize step
        viterbi_matrix = numpy.zeros((len(self.__tag_set), len(observations)), dtype =float)
        backpointer = numpy.zeros((len(self.__tag_set), len(observations)), dtype =int)
        for s in range(len(self.__tag_set)):
            viterbi_matrix[s, 0] = self.__start_probability[self.__tag_set[s]] * self.__emissions[observations[0]][s]

        #iteration step
        for o in range(1, len(observations)):
            for s in range(len(self.__tag_set)):
                prec = viterbi_matrix[:, o-1]  
                trans = self.__transitions[self.__tag_set[s]] 
                em = self.__emissions[observations[o]][s] 
                product = [prec[i] * trans[i] * em for i in range(len(prec))]
                maxIndex = numpy.argmax(product)
                viterbi_matrix[s,o] = viterbi_matrix[maxIndex,o-1] * self.__transitions[self.__tag_set[s]][maxIndex] * em
                backpointer[s, o] = maxIndex
    
        
        # final step
        # backtracking from last observation to the first in order
        bestPath = ["" for el in observations]
        finalState = [viterbi_matrix[i , len(observations)-1] * self.__final_probability[self.__tag_set[i]] for i in range(len(self.__tag_set))]
        maxIndex = numpy.argmax(finalState)

        for o in range(len(observations)-1, -1, -1):
            bestPath[o] = self.__tag_set[maxIndex]
            maxIndex = backpointer[maxIndex, o]

        return bestPath
        

    def baseline(self, observations, smoothing_tag):
        """
            Definition of a baseline to make a comparison with the viterbi algorithm
            This baseline assigns the most frequent tag in the training dataset, for unknown word assigns smoothing_tag
            
            Args:
                observations(list str): a list of a observations
                smoothing_tag(str): a tag used as smoothing strategy. If the observation-i is a unknown word, tag observation-i as smoothing_tag.  
                USAGE: ['O', 'B-MISC', 'B-ORG' ...]               

            Returns:
                best_path(list str): a list of a tag used to tagging the observation passed on args
        """
        
        if not smoothing_tag in self.__tag_set:   
            raise ValueError("Incorrect using of smoothing parameter!\n USAGE: ['O', 'B-MISC', 'B-ORG' ...]")

        best_path = []
        if not len(self.__baseline_tag) == 0:
            exclusive = [obs for obs in observations if not(obs in self.__baseline_tag)]
            for i in range(len(observations)):
                if observations[i] in exclusive:
                    best_path.insert(i, smoothing_tag)
                else:
                    tag = numpy.argmax(self.__baseline_tag[observations[i]])
                    best_path.insert(i, self.__tag_set[tag])
            return best_path
        
        
        occurrences = dict()
        for sent in observations:
            if not sent in occurrences:
                occurrences[sent] = [0 for el in self.__tag_set]

        # counts (word,tag) occurrences
        for line in self.__labelled_dataset:
            if line.split() != []:
                _, word, tag = line.split()
                if word in observations:
                    tag_index = self.__tag_set.index(tag)
                    occurrences[word][tag_index] += 1 

        # if observation[i] is unknown assigns B-MISC otherwise most frequent tag
        for i in range(len(observations)):
            tag_occurrences = occurrences[observations[i]]
            check = [el for el in tag_occurrences if el == 0]
            if len(tag_occurrences) == len(check):
                best_path.insert(i, smoothing_tag)
            else:
                tag = numpy.argmax(occurrences[observations[i]])
                best_path.insert(i, self.__tag_set[tag])
        
        return best_path


    def testing_viterbi(self, test_set_path, smoothing, number_sentence = None):
        """
            Testing HMM NER tagger with Viterbi's decoding algorithm on a testing dataset

            Args:
                test_set_path(str): path to a test dataset organize in CoNLL style
                smoothing(str): the declaration of a smoothing strategy for calculate unknown words emission probabilities.
                    USAGE: ['smoothing1', 'smoothing2', 'smoothing3'] or passing a file '.txt' dataset path"
                [Optionally] number_sentence(int): number of sentences to tagging. If not specified the algorithm analizes all the dataset

            Exceptions:
                ValueError exception: if dataset is incorrect formatting as CoNLL style
                ValueError exception: for incorrect smoothing parameter
                FileNotFound exception: in dataset not exists
        """

        try:
            dataset = self.__open_and_check(test_set_path)
        except Exception as e:
            print(e)
            exit()

        self.__reinitialize()
        sentences = []
        sentence = []
        for line in dataset:
            if line.split() != []:
                _, word, tag = line.split()
                sentence.append((word,tag))
            else:
                sentences.append(sentence)
                sentence = []
        
        n_sentences = (len(sentences)+1) if number_sentence == None else number_sentence
        
        for coupplesTagged in sentences[:number_sentence]:
            sentenceSplitted = [el[0] for el in coupplesTagged]
            best_path = self.viterbi(sentenceSplitted, smoothing)
            correct_best_path = [couple[1] for couple in coupplesTagged]
            self.__evaluation_measures(correct_best_path, best_path)

        # accurarcy, precision and recall
        self.__fill_metrics(n_sentences)
        

    def testing_baseline(self, test_set_path, smoothing_tag, number_sentence = None):
        """
            Testing of a simple baseline on a testing dataset

            Args:
                test_set_path(str): path to a test dataset organize in CoNLL style
                smoothing_tag(str): A tag used as smoothing strategy. If the observation-i is a unknown word, tag observation-i as smoothing_tag. USAGE: ['O', 'B-MISC', 'B-ORG' ...] 
                [Optionally] number_sentence(int): number of sentences to tagging. If not specified the algorithm analizes all the dataset
        """

        try:
            dataset = self.__open_and_check(test_set_path)
        except Exception as e:
            print(e)
            exit()
            
        self.__reinitialize()
        sentences = []
        sentence = []
        for line in dataset:
            if line.split() != []:
                _, word, tag = line.split()
                sentence.append((word,tag))
            else:
                sentences.append(sentence)
                sentence = []

        n_sentences = (len(sentences)+1) if number_sentence == None else number_sentence
        
        for line in self.__labelled_dataset:
            if line.split() != []:
                _, word, tag = line.split()
                tag_index = self.__tag_set.index(tag)
                if word not in self.__baseline_tag:
                    self.__baseline_tag[word] = [0 for el in self.__tag_set]
                    self.__baseline_tag[word][tag_index] = 1
                else:
                    self.__baseline_tag[word][tag_index] += 1

        for coupplesTagged in sentences[:number_sentence]:
            sentence_splitted = [el[0] for el in coupplesTagged]

            try: best_path = self.baseline(sentence_splitted, smoothing_tag)
            except Exception as e:
                print(e)
                exit()
                
            correct_best_path = [couple[1] for couple in coupplesTagged]
            self.__evaluation_measures(correct_best_path, best_path)
        
        self.__fill_metrics(n_sentences)



    # Private methods
    def __learning(self, dataset_path):
        """
            Performs the learning on a training dataset organized in CoNLL style
            This function counts the sentences number, retrieves the tagset and calculate all transition, initial and final probabilities

            Args:
                dataset_path(str): the path of a training dataset uset to perform the learning 
            
            Exception:
                ValueError exception: if dataset is incorrect formatting as CoNLL style
                FileNotFound exception: in dataset not exists
        """
        
        try:
            self.__labelled_dataset = self.__open_and_check(dataset_path)
        except Exception as e:
            print(e)
            exit()
               
        for i in range(len(self.__labelled_dataset)):        
            
            if self.__labelled_dataset[i].split() != []:
                if i == len(self.__labelled_dataset)-1:
                    id, _, tag = self.__labelled_dataset[i].split()
                    if tag not in self.__tag_final_occurrences:
                        self.__tag_final_occurrences[tag] = 1
                    else:
                        self.__tag_final_occurrences[tag] += 1

                id, _, tag = self.__labelled_dataset[i].split()
                if id == "0":
                    self.__sentence_number += 1
                    if tag not in self.__tag_initial_occurrences:
                        self.__tag_initial_occurrences[tag] = 1
                    else:
                        self.__tag_initial_occurrences[tag] += 1
           
                if tag not in self.__tag_set:
                    self.__tag_set.append(tag)
                if tag not in self.__tag_occurrences:
                    self.__tag_occurrences[tag] = 1
                else:
                    self.__tag_occurrences[tag] += 1       
            else:
                id, _, tag = self.__labelled_dataset[i-1].split()
                if tag not in self.__tag_final_occurrences:
                    self.__tag_final_occurrences[tag] = 1
                else:
                    self.__tag_final_occurrences[tag] += 1
        
        self.__transition_probabilities_matrix()
        self.__initial_probability()
        self.__end_probability()


    def __open_and_check(self, dataset_path):
        """
            Open a file organize in CoNLL style
            
            Args:
                dataset_path(str): path of a file

            Exception:
                throw FileNotFoundError if file not exists
        """
        try:
            dataset = open(dataset_path, "r", encoding='utf8')
        except FileNotFoundError:
            print("Error! File not found...")
            exit()

        dataset_lines = dataset.readlines()
        if dataset_lines == []:
            raise ValueError("Error, void dataSet in input!")

        if len(dataset_lines[0].split()) != 3:
            raise ValueError("Error reading dataSet fields.\nDataSet must have 3 fields!\nAccepted dataSet has this form: ID, WORD, TAG")

        dataset.close()
        return dataset_lines


    def __initial_probability(self):
        """
            This function calculates all initial probability P(TAG|START) for all TAG in tagset as:
            count(tag_initial_occurrences) / Count(sentence_number) 
        """
        for tag in self.__tag_set:
            if tag not in self.__start_probability:
                if tag not in self.__tag_initial_occurrences:
                    self.__start_probability[tag] = self.__transition_smoothing
                else:
                    self.__start_probability[tag] = self.__tag_initial_occurrences[tag] / self.__sentence_number


    def __end_probability(self):
        """
            This function calculates all final probability P(END|TAG) for all TAG in tagset as:
            count(tag_final_occurrences) / count(tag_occurrences)
        """
        for tag in self.__tag_set:
            if tag not in self.__final_probability:
                if tag not in self.__tag_final_occurrences:
                    self.__final_probability[tag] = self.__transition_smoothing
                else:
                    self.__final_probability[tag] = self.__tag_final_occurrences[tag] / self.__tag_occurrences[tag]


    def __transition_probabilities_matrix(self):
        """
            This function calculates all transition probabilities P(TAG|TAGi) for all TAG and i in range(tagset)
        """
        tagset = self.__tag_set
        for tag in tagset:
            if not tag in self.__transitions:
                self.__transitions[tag] = [0 for el in tagset]

        # counts for all sentence the tag1,tag2 occurrences
        for i in range(len(self.__labelled_dataset)-1):
            if(self.__labelled_dataset[i].split() != [] and self.__labelled_dataset[i+1].split() != []):
                _, _, tag1 = self.__labelled_dataset[i].split()
                _, _, tag2 = self.__labelled_dataset[i+1].split()
                tag1_index = tagset.index(tag1)
                self.__transitions[tag2][tag1_index] += 1
                
        # divides all occurrences tag1,tag2 by tag1 occurrences
        for tag in self.__transitions:
            for i in range(len(self.__transitions[tag])):
                if(self.__transitions[tag][i] == 0):
                    self.__transitions[tag][i] = self.__transition_smoothing
                else:
                    self.__transitions[tag][i] = self.__transitions[tag][i]/ self.__tag_occurrences[tagset[i]]    

        # la somma delle P(TAGi | TAG2) per ogni i e fissato TAG2 è uguale a 1
        

    def __all_emission_probability(self, observations, smoothing):
        """
            This function calculates all the emission probabilities P(observations_i|TAGj) for all tag
            and for all observations in the training dataset

            Args:
                observations(list str): a list of a observations
                smoothing(str): the declaration of a smoothing strategy for calculate unknown words emission probabilities
                validation_path(str): the path of a validation set to perform the smoothing4 strategy
        """

        if not smoothing in ["smoothing1", "smoothing2", "smoothing3"]:
            check_path = re.search(".*.txt$", smoothing)
            if not check_path:          
                raise ValueError("Incorrect using of smoothing parameter!\n USAGE: ['smoothing1', 'smoothing2', 'smoothing3'] or passing a file .txt path")


        tagset = self.__tag_set
        if len(self.__emissions) == 0:
            for line in self.__labelled_dataset:
                if line.split() != []:
                    _, word, tag = line.split()
                    tag_index = tagset.index(tag)
                    if word not in self.__emissions:
                        self.__emissions[word] = [0 for el in tagset]
                        self.__emissions[word][tag_index] = 1
                    else:
                        self.__emissions[word][tag_index] += 1
        
            for word in self.__emissions:
                for i in range(len(self.__emissions[word])):
                    self.__emissions[word][i] = self.__emissions[word][i] / self.__tag_occurrences[tagset[i]]

        # list of unknown words
        exclusive = [sent for sent in observations if not sent in self.__emissions]
        if len(exclusive) == 0:
            return
        
        #smoothing
        check_path = re.search(".*.txt$", smoothing)
        if check_path and self.__statistics == {}:
            self.__statistics = self.__smoothing4(smoothing)

        for sent in exclusive:    
            self.__emissions.update({sent:[0 for el in tagset]})
            for i in range(len(self.__emissions[sent])):
                if smoothing == "smoothing1":
                    self.__emissions[sent][i] = 1 if tagset[i] == "O" else 0
                elif smoothing == "smoothing2":
                    self.__emissions[sent][i] = 0.5 if tagset[i] in ["O", "B-MISC"] else 0
                elif smoothing == "smoothing3":
                    self.__emissions[sent][i] = 1 / len(tagset)
                elif check_path:
                    self.__emissions[sent][i] = self.__statistics["tagCount"][i] / self.__statistics["length"]


    def __smoothing4(self, validation_path):
        """
            This function performs a smoothing strategy for calculate the emission probabilities for the unknown words.
            Assume the unknown words have a probability distribution similar to words only occurring once in the validation set.
            Returns an unknown object(tagcount:int, length:int)
        """

        try:
            datasetLines = self.__open_and_check(validation_path)
        except Exception as e:
            print(e)
            exit()
            
        occurrences = dict()
        for line in datasetLines:
            if line.split() != []:
                _, word, tag = line.split()
                if not word in occurrences:
                    occurrences[word] = tag
                else:
                    occurrences[word] = "No"
        
        tagCount = [0 for el in self.__tag_set]
        for word in occurrences:
            if occurrences[word] != "No":
                tagCount[self.__tag_set.index(occurrences[word])] += 1
        
        return {"tagCount":tagCount, "length":sum(tagCount)}
      

    def __evaluation_measures(self, correct_best_path, best_path):
        """
            Calculates accuracy, precision and recall evaluations metrics, from the test_set analysis

            Args:
                coorrect_best_path(list str): list of tag corresponding to the correct tags
                best_path(list str): list of tag tagged by the NER
        """

        # for general accuracy:
        check = len([i for i in range(len(correct_best_path)) if correct_best_path[i] == best_path[i]])
        self.__accuracy["GENERAL"]["ACCURACY"] += (check / len(correct_best_path))
        
        # for accuracy of tag O
        total_O = len([i for i in range(len(correct_best_path)) if correct_best_path[i] == "O"])
        check = len([i for i in range(len(correct_best_path)) if (correct_best_path[i] == best_path[i]) and correct_best_path[i] == "O"])
        self.__accuracy["O"]["COR"] += check
        self.__accuracy["O"]["TOT"] += total_O

        # for accuracy, precision and recall on entities:
        # for each TAG it calculates the NER entities both on the original sentence and on the sentence analyzed by the tagger
        self.__metrics_per_tag_entity(correct_best_path, best_path, "ORG", "B-ORG", "I-ORG")
        self.__metrics_per_tag_entity(correct_best_path, best_path, "PER", "B-PER", "I-PER")
        self.__metrics_per_tag_entity(correct_best_path, best_path, "MISC", "B-MISC", "I-MISC")
        self.__metrics_per_tag_entity(correct_best_path, best_path, "LOC", "B-LOC", "I-LOC")

    
    def __metrics_per_tag_entity(self, correct_best_path, best_path, tag_entity, tagB, tagI):
        """
            For each (B-TAG, I-TAG) it calculates the NER entities 
            both on the original sentence and on the sentence analyzed by the tagger
    
            Args:
                coorrect_best_path(list str): list of tag corresponding to the correct tags
                best_path(list str): list of tag tagged by the NER
                tag_entity(str): normal form of a entity tag
                tagB (str): a B-TAG
                tagI (str): a I-TAG
        """


        # computing accuracy per ENTITY:   
        totals = len([i for i in range(len(correct_best_path)) if (correct_best_path[i] == tagB) or correct_best_path[i] == tagI])
        self.__accuracy[tag_entity]["TOT"] += totals

        for i in range(len(correct_best_path)):
            if((correct_best_path[i] == tagB) and (best_path[i] == tagB)) or ((correct_best_path[i] == tagI) and (best_path[i] == tagI)):
                self.__accuracy[tag_entity]["COR"] += 1

        
        entity = []                 # all the entity in the correct list of tag
        tagged_entity = []          # all the entity tagged by the NER tagger

        for i in range(len(correct_best_path)):
            if correct_best_path[i] == tagB:
                j = i+1
                check = False
                while j < len(correct_best_path) and correct_best_path[j] == tagI:
                    j += 1
                    check = True
                if check: 
                    entity.append((tagB, i, j-1))
                    i = j
                else:
                    entity.append((tagB, i, i))

        i = 0
        while (i < len(best_path)):
            if best_path[i] == tagB:
                j = i+1
                check = False
                while j < len(best_path) and best_path[j] == tagI:
                    j += 1
                    check = True
                if check: 
                    tagged_entity.append((tagB, i, j-1))
                    i = j-1
                else:
                    tagged_entity.append((tagB, i, i))
            i+=1


        # calculates true positives as all those correctly labeled entities
        # calculate false negatives as all those entities not tagged by the NER tagger (tag O)  
        for el in entity:
            if el in tagged_entity: self.__metrics[tag_entity]["COR"] += 1
            else: self.__metrics[tag_entity]["MIS"] += 1
        
        # calculates false positives such as tagged entities
        #   as positive but they were actually negative
        for el in tagged_entity:
            if not el in entity: self.__metrics[tag_entity]["HYP"] += 1


    def __fill_metrics(self, number_sentences):
        """
            Utility function, used to fill the evaluation metric dictionaries 
        """


        # general accurarcy 
        self.__accuracy["GENERAL"]["ACCURACY"] = round(100 * (self.__accuracy["GENERAL"]["ACCURACY"] / number_sentences),3)
        self.__accuracy["O"]["ACCURACY"] = round(100 * (self.__accuracy["O"]["COR"] / self.__accuracy["O"]["TOT"]),3)

        # local accuracy, precision, recall, f1score
        for entity in self.__metrics:
            if entity != "O":

                metric = self.__metrics[entity]
                if((metric["COR"] + metric["HYP"]) > 0):
                    precision = round(100 * (metric["COR"] / (metric["COR"] + metric["HYP"])),3)
                    recall = round(100 * (metric["COR"] / (metric["COR"] + metric["MIS"])),3)
                    self.__metrics[entity]["PRECISION"] = precision
                    self.__metrics[entity]["RECALL"] = recall
                    self.__metrics[entity]["F1SCORE"] = round((2 * precision * recall) / (precision + recall),3)
                
                if (self.__accuracy[entity]["TOT"] > 0):
                    entity_accuracy = round(100 * (self.__accuracy[entity]["COR"] / self.__accuracy[entity]["TOT"]), 3)
                
                self.__accuracy[entity]["ACCURACY"] = entity_accuracy


    def __reinitialize(self):
        """
            Reinitialize all evaluating metrics for a second analysis
        """
        self.__emissions = dict()
        self.__accuracy = 0
        self.__metrics = {"ORG": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "PER": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "MISC": {"COR":0, "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}, 
                        "LOC": {"COR":0,  "MIS":0, "HYP":0, "PRECISION": 0, "RECALL": 0, "F1SCORE": 0}}
        
        self.__accuracy = { "GENERAL": {"ACCURACY":0},
                            "ORG": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "PER": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "MISC": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "LOC": {"TOT": 0, "COR": 0 ,"ACCURACY":0},
                            "O": {"TOT": 0, "COR": 0 , "ACCURACY":0}}   