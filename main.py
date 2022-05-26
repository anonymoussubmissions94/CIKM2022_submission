from nltk.tokenize import sent_tokenize, word_tokenize  
from numpy import number
import os
#from t2e import predict_events, predict_sentence_events
from classes import * 
from qa import run_QA
import ndjson
from utils import * 
from random import shuffle
from collections import Counter,  OrderedDict
import numpy as np
import configparser
from nltk.tokenize.treebank import TreebankWordDetokenizer
import json
import spacy
nlp = spacy.blank("en")
nlp.add_pipe('opentapioca')
types_shelve = shelve.open('subevents/shelves/instance_types')
dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')
X = dict()
with open("arg-prop_mapping.csv", "r") as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        X[row[0]] = (row[1], row[2])

config = configparser.ConfigParser()
config.read("config.ini")
project_path = config.get("Paths","project_path")
data_path = config.get("Paths","data_path")
prediction_path = config.get("Paths","prediction_path")

def create_transitive_class_hierarchy():
    d = dict()
    hd = dict()
    with open('subevents/data/class_parents.csv') as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data:  
            child = row[0]  
            parent = row[1]
            if child not in d:
                d[child] = []
            d[child].append(parent)
    print(len(d))
    for i, child in enumerate(d):
        if child not in hd:
            hd[child] = []
        hd[child] = get_parents(d, child, [])
        if i%10000==0:
            print(i, len(d))
    with open("subevents/data/transitive_class_hierarchy.json","w") as f:
        json.dump(hd, f)

def get_parents(parent_map, node_id, filter):
    if node_id in parent_map:
        parents = set(parent_map[node_id])
        for parent_id in parent_map[node_id]:
            if parent_id in filter:
                continue
            filter.append(parent_id)
            parents |= set(get_parents(parent_map, parent_id, filter))
    else:
        parents = set()
    return list(parents)

def predict_events(prediction_path, data_path, stop_at=150):
    last_time = datetime.datetime(2001,12,10)
    print("START")
    directory = prediction_path
    s = 0
    l = 0
    for subdirectory in sorted(os.listdir(directory)):
        print(subdirectory)
        dataset  = {}
        all_data_path = data_path
        all_files = (sorted(os.listdir(all_data_path)))
        with open(all_data_path + all_files[int(subdirectory)], "r") as f:
            json_list = ndjson.load(f)
        for number_of_processed_articles, article in enumerate(sorted(os.listdir(directory+subdirectory))): 
            if (article.endswith("_1.txt") or article.endswith("_2.txt") or article.endswith("_3.txt") or article.endswith("_4.txt")):
                continue
            if l > 0 and l-s>=2:
                print("Dumping data\n\n\n")
                with open("data/unlinked_events/a_"+str(l)+".json", "w") as fx:
                    json.dump(dataset, fx, default=str)
                dataset = {}
                s = l
            print(article)
            with open(directory+subdirectory+"/"+article,"r") as f:
                predictions = json.load(f)
            if not predictions or "_FAILED_PREDICTION" in predictions:
                continue 
            Article_obj = Article(json_list[int(article.split(".")[0])])
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
            except (KeyError):
                continue

            print("number_of_processed_articles: ", l)
            l+=1
            if l>stop_at:
                break

            sentences = predictions["sentences"]
            all_predicted_events = predictions["predicted_events"]
            dataset[parent_event_wd_id] = {}

            all_tokens = [token for sentence in sentences for token in sentence]    

            current_context_window = 0
            event_types = []
            event_triggers = {}
            all_sentences_detokenized = [TreebankWordDetokenizer().detokenize(sentence) for sentence in sentences]
            detokenized_sentences = []

            for i in range(len(all_sentences_detokenized)):
                if all_predicted_events[i]:
                #if  predicted_events_by_sentence_index[i]:
                    current_context_window += 1 
                    detokenized_sentences.append(all_sentences_detokenized[i])
                    for prediction in all_predicted_events[i]:
                    #for prediction in  predicted_events_by_sentence_index[i]:
                        event_type = prediction[0][1].split(".")[1].upper() # [[[14, "Conflict.Attack", 28.494, 1.0]
                        event_trigger = all_tokens[prediction[0][0]] 
                        event_types.append(event_type)
                        if event_type not in event_triggers:
                            event_triggers[event_type] = []
                        event_triggers[event_type].append(event_trigger)

                    if len(all_sentences_detokenized)>i+1:
                        future_predicted_events = all_predicted_events[i+1]
                        #future_predicted_events =  predicted_events_by_sentence_index[i+1]
                        future_predicted_event_types = [prediction[0][1].split(".")[1].upper() for prediction in future_predicted_events]
                        if (set(future_predicted_event_types) & set(event_types)) or current_context_window==2:
                            detokenized_sentences = " ".join(detokenized_sentences)
                            dataset[parent_event_wd_id][detokenized_sentences], last_time= extract_event(detokenized_sentences, event_types, event_triggers, last_time)
                            detokenized_sentences = []
                            event_types = []
                            event_triggers = {}
                            current_context_window = 0
                        else:
                            continue
                    else:
                        detokenized_sentences = " ".join(detokenized_sentences)
                        dataset[parent_event_wd_id][detokenized_sentences], last_time= extract_event(detokenized_sentences, event_types, event_triggers, last_time)
                        detokenized_sentences = []
                        event_types = []
                        event_triggers = {}
                        current_context_window = 0


def extract_event(detokenized_sentences, event_types, event_triggers, last_time):
    completed = {}
    filter = []
    context = detokenized_sentences
    initial_candidates_lists = [get_initial_candidates(event_type) for event_type in event_types if get_initial_candidates(event_type)]
    new_candidates = initial_candidates_lists[:]
    questions, indices, constraints = generate_questions(initial_candidates_lists, completed)
    all_candidates = []
    candidate_arguments = {} # informal_arguments = {}
    while questions:
        qa_answers = run_QA(questions, context)
        qa_answers = [pair if pair and pair[1]>0.0 else 0 for pair in qa_answers]
        only_probs = [pair[1] if pair and pair[1]>0.0 else 0 for pair in qa_answers]
        only_answers = [pair[0] if pair and pair[1]>0.0 else 0 for pair in qa_answers]
        new_candidates = []
        temp = dict()
        
        for start,end in indices:
            # extract candidate, property, event type for each question - answer pair
            candidate = indices[(start,end)][0]  
            property = indices[(start,end)][1]  
            event_type = indices[(start,end)][2]  
            # remember candidate-property pairs that we have found answers for
            if property not in completed:
                completed[property] = []
            completed[property].append(candidate)
            all_candidates.append((candidate,event_type))
            if not any(qa_answers[start:end]):
                continue
            # find best scores for a specific candidate-property pair
            best_answer_score_indices =  np.flatnonzero(only_probs[start:end] == np.max(only_probs[start:end])).tolist()
            best_answer_score_indices = [el + start for el in best_answer_score_indices]
            best_answer_score = only_probs[best_answer_score_indices[0]]
            best_answers = [only_answers[index] for index in best_answer_score_indices]
            if event_type not in temp:
                temp[event_type] = {property:[]}
            temp[event_type][property].append((candidate, best_answer_score, best_answers))

        # create a preliminary candidate-argument dictionary for all candidate-property pairs such that it containst only max probability candidate-property pairs, 
        # and max probability property-answer pairs
        preliminary_candidates = {}
        for event_type in temp:
            for property in temp[event_type]:
                # for each event-type - property pair, find those with the best scores and get indices of such candidate,best_answer_score, best_answers tuples.
                # Then store each of these "best" tuples into a dictionary. 
                all_property_scores = [best_answer_score for candidate,best_answer_score, best_answers in temp[event_type][property]]
                best_candidate_for_property_score_indices =  np.flatnonzero(all_property_scores == np.max(all_property_scores)).tolist()
                for i in best_candidate_for_property_score_indices:
                    if event_type not in preliminary_candidates:
                        preliminary_candidates[event_type]={property:[]}
                    cand = temp[event_type][property][i][0]
                    best_answer_score = temp[event_type][property][i][1]
                    best_answers = temp[event_type][property][i][2]
                    preliminary_candidates[event_type][property].append((cand,best_answer_score, best_answers))
        # check our candidate arguments against constraints           
        for event_type in preliminary_candidates:
            for property in preliminary_candidates[event_type]:
                for candidate_answer_pair in preliminary_candidates[event_type][property]:
                    candidate = candidate_answer_pair[0]
                    best_answer_score = candidate_answer_pair[1]
                    best_answers = candidate_answer_pair[2]
                    try:
                        argument_constraints = list(set(prop2constraints[candidate][property]))
                    except KeyError:
                        with open("constraints_not_founds.txt","a") as f:
                            f.write(candidate + "\t" + property + "\n")
                        argument_constraints = []
                    arguments = [Argument(property, best_answers, argument_constraints, last_time)]
                    #predicted_ner_texts = [" ".join(all_tokens[ne[0]:ne[1]+1]) for ne in predicted_ner]
                    arguments = assign_ner(arguments)
                    arguments = [(argument, best_answer_score) for argument in arguments if argument]
                    
                    if arguments:
                        last_time = arguments[-1][0].last_time
                        if arguments[0][0].type=="TIMEX":
                            argument = select_most_precise_time_expression(arguments)
                        else:
                            argument = sorted(arguments, key=lambda x: x[1], reverse=True)[0]

                        if (candidate, event_type) not in candidate_arguments:
                            candidate_arguments[(candidate,event_type)] = dict() # informal_arguments[(candidate,event_type)] = dict()

                        if not argument:
                            continue
                        else:
                            candidate_arguments[(candidate,event_type)][property] = argument   
                            new_candidates.append((candidate,event_type)) 
                        # passing off superclass arguments to candidate arguments
                        if candidate in all_classes_superclasses:
                            for super_class in all_classes_superclasses:
                                if (super_class, event_type) in candidate_arguments:
                                    for property in candidate_arguments[(super_class, event_type)]:
                                        # only pass off arguments that have not already been assigned to the child candidate
                                        if property not in candidate_arguments[(candidate, event_type)]:
                                            candidate_arguments[(candidate, event_type)][property] = candidate_arguments[(super_class, event_type)][property] 
                                            completed[property].append(candidate)
        # remember all the candidates of event_types that didn't get an answer this round of questions
        #if new_candidates:
        if preliminary_candidates:
            # types_with_answers = [pair[1] for pair in new_candidates]
            types_with_answers = list(preliminary_candidates.keys())
            # filter out candidates that did not get an answer of event types that did get an answer
            #filter += [pair for pair in all_candidates if pair[1] in types_with_answers and pair not in new_candidates]
            filter += [pair for pair in all_candidates if pair[1] in types_with_answers and pair[0] not in [class_answer[0] for property in preliminary_candidates[pair[1]] for class_answer in preliminary_candidates[pair[1]][property]]]
            filter = list(set(filter))
            all_candidates = list(set([item for item in all_candidates if item not in filter]))
            #new_candidates = preliminary_candidates
            #leftover_candidates = [pair for pair in all_candidates if pair[1] not in types_with_answers]
        new_candidates = list(set(new_candidates))
        new_candidates, candidate_arguments, filter = get_new_candidates(new_candidates, candidate_arguments, filter)

        new_candidates+=all_candidates
        new_candidates = list(set(new_candidates))
        print()
        questions, indices, constraints = generate_questions(new_candidates, completed)
    results = identify_correct_classes(candidate_arguments, new_candidates, context)
    output = []

    if results:
        for result in results:
            wd_class = result[0][0]
            event_type = result[1]
            event_trigger = event_triggers[event_type]
            predicted_event_properties = {k:v[0].wikidata_id if v[0].wikidata_id else v[0].texts for k,v in candidate_arguments[(wd_class, event_type)].items()}
            output.append({"predicted_event_class":wd_class, "predicted_event_properties": predicted_event_properties, "event_trigger": event_trigger})
    
    return output, last_time


def reformat_t2e_predictions(predictions):
    all_predictions = []
    if not isinstance(predictions, list):
        predictions = [predictions]
    for prediction in predictions:
        formatted_prediction = {}
        roles = prediction["roles"]
        event_type = prediction["type"].upper()
        event_trigger = prediction["trigger"]
        argument_classes = []
        argument_properties = []
        if event_type not in X:
            continue
        for role in roles:
            role_type = role[1]
            role_text = role[2]
            for span in nlp(role_text).ents:
                if span.kb_id_ in types_shelve:
                    if event_type not in X or role_type not in eval(X[event_type][1]):
                        continue
                    argument_classes.append(span.kb_id_)
                    argument_properties.append(eval(X[event_type][1])[role_type])
        wd_class = X[event_type][0]
        formatted_prediction = {"predicted_event_class": wd_class, "predicted_event_properties":{}}
        formatted_prediction["predicted_event_properties"] = {k:v for k,v in zip(argument_properties, argument_classes)}
        all_predictions.append(formatted_prediction)
    return all_predictions


def ace2wd(event, arguments, all_tokens):
    event_type = event[1].split(".")[1].upper()
    if event_type not in X:
        return False
    argument_texts = [" ".join(all_tokens[argument[0]:argument[1]+1]) for argument in arguments]
    argument_types = [argument[2] for argument in arguments]
    argument_classes = []
    argument_properties = []
    for i in range(len(argument_texts)):
        for span in nlp(argument_texts[i]).ents:
            if span.kb_id_ in types_shelve:
                if event_type not in X or argument_types[i] not in eval(X[event_type][1]):
                    continue
                argument_classes.append(span.kb_id_)
                argument_properties.append(eval(X[event_type][1])[argument_types[i]])
    wd_class = X[event_type][0]
    prediction = {"predicted_event_class":wd_class,"predicted_event_properties":{}}
    prediction["predicted_event_properties"] = {k:v for k,v in zip(argument_properties, argument_classes)}
    return prediction


def t2e(predictions_path, data_path, stop_at=150):
    print("START")
    directory = predictions_path
    s = 0
    l = 0
    for subdirectory in sorted(os.listdir(directory)):
        print(subdirectory)
        dataset  = {}
        all_data_path = data_path
        all_files = (sorted(os.listdir(all_data_path)))
        with open(all_data_path + all_files[int(subdirectory)], "r") as f:
            json_list = ndjson.load(f)
        for number_of_processed_articles, article in enumerate(sorted(os.listdir(directory+subdirectory))): 
            if (article.endswith("_1.txt") or article.endswith("_2.txt") or article.endswith("_3.txt") or article.endswith("_4.txt")):
                continue
            if l > 0 and l-s>=2:
                print("Dumping data\n\n\n")
                with open("data/unlinked_events/t2e_"+str(l)+".json", "w") as fx:
                    json.dump(dataset, fx, default=str)
                dataset = {}
                s = l
            print(article)
            with open(directory+subdirectory+"/"+article,"r") as f:
                predictions = json.load(f)
            if not predictions or "_FAILED_PREDICTION" in predictions:
                continue 
            Article_obj = Article(json_list[int(article.split(".")[0])])
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
            except (KeyError):
                continue

            print("number_of_processed_articles: ", l)
            l+=1
            if l>stop_at:
                break
        
            sentences = predictions["sentences"]
            all_predicted_events = predictions["predicted_events"]
            dataset[parent_event_wd_id] = {}

            all_tokens = [token for sentence in sentences for token in sentence]    
            current_context_window = 0
            event_types = []
            event_triggers = {}
            all_sentences_detokenized = [TreebankWordDetokenizer().detokenize(sentence) for sentence in sentences]
            detokenized_sentences = []
            sentence_edges = {}
            tmp = 0
            for i, sentence in enumerate(sentences):
                sentence_edges[tmp+len(sentence)-1] = i
                tmp = tmp+len(sentence)
            predicted_events_by_sentence_index = {}
            for predicted_event in all_predicted_events:
                if not predicted_event:
                    continue
                trigger_index = predicted_event[0][0][0]
                for sentence_edge in sentence_edges:
                    if trigger_index<sentence_edge:
                        if sentence_edges[sentence_edge] not in predicted_events_by_sentence_index:
                            predicted_events_by_sentence_index[sentence_edges[sentence_edge]] = []
                        predicted_events_by_sentence_index[sentence_edges[sentence_edge]].append(predicted_event)
                        break
            #dataset[parent_event_wd_id][detokenized_sentences] = t2e(all_sentences_detokenized, predicted_events_by_sentence_index, dataset, parent_event_wd_id)
            for i in range(len(all_sentences_detokenized)):
                if all_predicted_events[i]:
                    current_context_window += 1 
                    detokenized_sentences.append(all_sentences_detokenized[i])
                    for prediction in all_predicted_events[i]:
                        event_type = prediction[0][1].split(".")[1].upper() # [[[14, "Conflict.Attack", 28.494, 1.0]
                        event_trigger = all_tokens[prediction[0][0]] 
                        event_types.append(event_type)
                        event_triggers[event_type] = event_trigger
                    if len(all_sentences_detokenized)>i+1:
                        future_predicted_events = all_predicted_events[i+1]
                        future_predicted_event_types = [prediction[0][1].split(".")[1].upper() for prediction in future_predicted_events]
                        if (set(future_predicted_event_types) & set(event_types)) or current_context_window==2:
                            detokenized_sentences = " ".join(detokenized_sentences)
                            t2e_predictions = predict_sentence_events([detokenized_sentences])
                            t2e_predictions  = reformat_t2e_predictions(t2e_predictions[0])
                            dataset[parent_event_wd_id][detokenized_sentences]= t2e_predictions
                            detokenized_sentences = []
                            event_types = []
                            event_triggers = {}
                            current_context_window = 0
                        else:
                            continue
                    else:
                        detokenized_sentences = " ".join(detokenized_sentences)
                        t2e_predictions = predict_sentence_events([detokenized_sentences])
                        t2e_predictions  = reformat_t2e_predictions(t2e_predictions[0])
                        dataset[parent_event_wd_id][detokenized_sentences]= t2e_predictions
                        detokenized_sentences = []
                        event_types = []
                        event_triggers = {}
                        current_context_window = 0


def reformat_dygiepp(predictions, all_tokens): 
    all_predictions = []
    for sentence_predictions in predictions:
        for prediction in sentence_predictions:
            event_prediction = prediction[0]
            argument_prediction = prediction[1:]
            prediction = ace2wd(event_prediction, argument_prediction, all_tokens)
            all_predictions.append(prediction)
    return all_predictions

def dygiepp(predictions, dataset, parent_event_wd_id):
    sentences = predictions["sentences"]
    all_predicted_events = predictions["predicted_events"]
    all_predicted_ner = predictions["predicted_ner"]
    dataset[parent_event_wd_id] = {}
    all_sentences_detokenized = [TreebankWordDetokenizer().detokenize(sentence) for sentence in sentences]
    all_tokens = [token for sentence in sentences for token in sentence]    
    predicted_ner = [ne for sentence_ne in all_predicted_ner for ne in sentence_ne]
    current_context_window = 0
    event_types = []
    event_triggers = {}
    context_ner = []
    detokenized_sentences = []
    context_predictions = []
    sentence_edges = {}
    tmp = 0
    for i, sentence in enumerate(sentences):
        sentence_edges[tmp+len(sentence)-1] = i
        tmp = tmp+len(sentence)
    predicted_events_by_sentence_index = {}
    for predicted_event in all_predicted_events:
        trigger_index = predicted_event[0][0][0]
        for sentence_edge in sentence_edges:
            if trigger_index<sentence_edge:
                if sentence_edges[sentence_edge] not in predicted_events_by_sentence_index:
                    predicted_events_by_sentence_index[sentence_edges[sentence_edge]] = []
                predicted_events_by_sentence_index[sentence_edges[sentence_edge]].append(predicted_event)
                break
    all_predicted_events = predicted_events_by_sentence_index
    for i in range(len(all_sentences_detokenized)):
        if all_predicted_events[i]:
            current_context_window += 1 
            detokenized_sentences.append(all_sentences_detokenized[i])
            for prediction in all_predicted_events[i]:
                event_type = prediction[0][1].split(".")[1].upper() 
                event_trigger = all_tokens[prediction[0][0]] 
                event_types.append(event_type)
                event_triggers[event_type] = event_trigger

            if len(all_sentences_detokenized)>i+1:
                future_predicted_events = all_predicted_events[i+1]
                future_predicted_event_types = [prediction[0][1].split(".")[1].upper() for prediction in future_predicted_events]
                context_predictions.append(prediction)
                context_ner.append(all_predicted_ner[i])
                if (set(future_predicted_event_types) & set(event_types)) or current_context_window==2:
                    detokenized_sentences = " ".join(detokenized_sentences)
                    dataset[parent_event_wd_id][detokenized_sentences]= reformat_dygiepp(context_predictions, all_tokens)
                    detokenized_sentences = []
                    event_types = []
                    context_predictions = []
                    current_context_window = 0
                else:
                    continue
            else:
                detokenized_sentences = " ".join(detokenized_sentences)
                dataset[parent_event_wd_id][detokenized_sentences]= reformat_dygiepp(context_predictions, predicted_ner)
                detokenized_sentences = []
                context_predictions = []
                event_types = []
                current_context_window = 0

    return dataset[parent_event_wd_id][detokenized_sentences]

def dygiepp_on_all(prediction_path, data_path, stop_at):
    print("START")
    directory = prediction_path
    s = 0
    l = 0
    tp = 0
    for subdirectory in sorted(os.listdir(directory)):
        print(subdirectory)
        dataset  = {}
        all_data_path = data_path
        all_files = (sorted(os.listdir(all_data_path)))
        with open(all_data_path + all_files[int(subdirectory)], "r") as f:
            json_list = ndjson.load(f)
        for number_of_processed_articles, article in enumerate(sorted(os.listdir(directory+subdirectory))): 
            if (article.endswith("_1.txt") or article.endswith("_2.txt") or article.endswith("_3.txt") or article.endswith("_4.txt")):
                continue
            if l > 0 and l-s>=2:
                print("Dumping data\n\n\n")
                with open("data/unlinked_events/dygiepp_ner_"+str(l)+".json", "w") as fx:
                    json.dump(dataset, fx, default=str)
                dataset = {}
                s = l
            print(article)
            with open(directory+subdirectory+"/"+article,"r") as f:
                predictions = json.load(f)
            if not predictions or "_FAILED_PREDICTION" in predictions:
                continue 
            Article_obj = Article(json_list[int(article.split(".")[0])])
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
            except (KeyError):
                continue
            print("number_of_processed_articles: ", l)
            l+=1
            if l>stop_at:
                break
            dataset.update(dygiepp(predictions,dataset, parent_event_wd_id))

            

if __name__ == "__main__":
    predict_events(prediction_path, data_path)
    t2e(prediction_path, data_path)
    dygiepp_on_all(prediction_path, data_path)
    create_table4()


    