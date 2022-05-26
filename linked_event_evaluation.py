import sys, os
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize  
from numpy import number
from t2e import predict_events, predict_sentence_events
import numpy as np
import json
from random import shuffle
from collections import Counter,  OrderedDict
import shelve
from nltk.tokenize.treebank import TreebankWordDetokenizer
from main import *
import datetime
from difflib import SequenceMatcher
import score
last_time = datetime.datetime(2001,12,10)


with open("evaluation/ground.txt", "r") as f:
    dataset = json.load(f)


# iterate through wikipedia articles
def our_approach(detokenized_sentence, event_types, event_triggers, last_time): 
    result_tple = extract_event(detokenized_sentence, event_types, event_triggers, last_time)
    data = result_tple[0]
    last_time = result_tple[1]
    return data, last_time #  [{"predicted_event_class":wd_class, \
    #"predicted_event_properties": predicted_event_properties, "event_trigger": event_trigger})

def t2e_baseline(t2e_predictions):
    return reformat_t2e_predictions(t2e_predictions) #[{"predicted_event_class": wd_class, "predicted_event_properties":{}}]


def dygiepp_baseline(predictions, all_tokens):
    return reformat_dygiepp(predictions, all_tokens) #[{"predicted_event_class": wd_class, "predicted_event_properties":{}}]

dygiepp_dataset = {}
t2e_dataset = {}
our_dataset = {}
count = 0
new_ground = {}

def run():
    for context_sentence, dygiepp_folder  in zip(list(dataset.keys()), sorted(os.listdir("evaluation/linked_sub-events/prediction"), key=lambda z: int(z))):
        #print(event)
        #if count<=190:
            #count+=1
            #continue
        if context_sentence not in new_ground:
            new_ground[context_sentence] = {}
        new_ground[context_sentence]["ground"] = dataset[context_sentence]
        with open("evaluation/linked_sub-events/prediction/"+dygiepp_folder+"/"+dygiepp_folder+".txt","r") as f:
            dygiepp_predictions = json.load(f)
        sentences = dygiepp_predictions["sentences"]
        dygiepp_predicted_events = dygiepp_predictions["predicted_events"]
        
        # iterate through subevents found in a wikipedia article
        if "dygiepp" not in dataset:
            new_ground[context_sentence]["dygiepp"] = []
        if "t2e" not in dataset:
            new_ground[context_sentence]["t2e"] = []
        if "our" not in dataset:
            new_ground[context_sentence]["our"] = []

        event_triggers = {}
        event_types = []
        all_tokens = [token for sentence in sentences for token in sentence]    
        for prediction in dygiepp_predicted_events:
            if prediction:
                trigger_index = prediction[0][0][0]
                event_type = prediction[0][0][1].split(".")[1].upper()
                event_types.append(event_type)
                if event_type not in event_triggers:
                    event_triggers[event_type] = []
                event_triggers[event_type].append(all_tokens[trigger_index])

        

        t2e_predicted_events = predict_sentence_events([context_sentence])
        if t2e_predicted_events:
            t2e_predicted_events_ = t2e_predicted_events[0]
        if t2e_predicted_events_:
            new_ground[context_sentence]["t2e"] = t2e_baseline(t2e_predicted_events_)
        new_ground[context_sentence]["dygiepp"]= dygiepp_baseline(dygiepp_predicted_events, all_tokens)
        new_ground[context_sentence]["our"], last_time = our_approach(context_sentence, event_types, event_triggers, last_time)

        count+=1
        if count%10==0:
            print("writing new batch")
            with open("evaluation/linked_sub-events/automatic/f_dataset_"+str(count)+".json","w") as f:
                json.dump(new_ground, f, default=str)
            new_ground = {}

if __name__ == "__main__":
    run()
    directory = "evaluation/linked_sub-events/automatic/"
    data = {}
    for i, d in enumerate(os.listdir(directory)):
        if d.startswith("f"):
            with open(directory+d, "r") as f:
                data.update(json.load(f))

    score.hp(data)
