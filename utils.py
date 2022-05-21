import csv
from doctest import ELLIPSIS_MARKER
import re
from dateparser import *
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy import number
from collections import Counter,  OrderedDict
import itertools
import spacy
import json
nlp = spacy.blank("en")
nlp.add_pipe('opentapioca')


ddp = DateDataParser()
all_classes_subclasses = dict()
all_classes_superclasses = dict()


import pandas as pd
import csv
import shelve
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
processing_sheets_path = config.get("Paths","processing_sheets_path")

dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')
types_dbo = shelve.open('subevents/shelves/types_dbo')
types_wd = shelve.open('subevents/shelves/types_wd')
types_shelve = shelve.open('subevents/shelves/instance_types')

with open("subevents/data/transitive_class_hierarchy.json","r") as f:
    class_hierarchy = json.load(f)



def assign_ner(arguments):
    #input: 
    # arguments: extracted argument objects
    # predicted_ner: predicted named entities in the span of the sentences from which the arguments where extracted
    for argument in arguments:
        for argument_text in argument.texts:
            doc = nlp(argument_text)
            for span in doc.ents:
                argument.wikidata_id.append(span.kb_id_)
                if span.kb_id_ in types_shelve:
                    argument.wikidata_types += types_shelve[span.kb_id_]
                elif span.kb_id_ in class_hierarchy:
                    argument.wikidata_types += [span.kb_id_]
    if not argument.wikidata_id: 
        #return arguments
        if len(argument_text.split(" "))==1:
            if argument_text in dbpedia_ids_to_wikidata_ids and argument_text in types_wd:
                argument.wikidata_id = dbpedia_ids_to_wikidata_ids[argument_text]
                argument.wikidata_types = types_wd[argument_text]
        elif len(argument_text.split(" "))>1:
            text = "_".join(argument_text.split(" "))
            if text in dbpedia_ids_to_wikidata_ids and text in types_wd:
                argument.wikidata_id = dbpedia_ids_to_wikidata_ids[text]
                argument.wikidata_types = types_wd[text]
    return arguments
    



with open(processing_sheets_path+"/all_classes_subclasses.csv", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        if row[0] not in all_classes_subclasses:
            all_classes_subclasses[row[0]] = []
        all_classes_subclasses[row[0]].append(row[1]) 

for superclass in all_classes_subclasses:
    subclasses = all_classes_subclasses[superclass]
    for subclass in subclasses:
        if subclass not in all_classes_superclasses:
            all_classes_superclasses[subclass] = []
        all_classes_superclasses[subclass].append(superclass)

def isDate(text, last_time):
    X= text.split(" ")
    t = ddp.get_date_tuple(text)
    date = None
    if t.date_obj:
        period = t.period
        if period == "day":
            date = (t.date_obj.year, t.date_obj.month, t.date_obj.day)
        elif period == "month":
            date = (t.date_obj.year, t.date_obj.month)
        elif period == "year":
            date = (t.date_obj.year)
    if any(X) and date==None:
        date = isDate(" ".join(X[1:]))
    return date

ace2classes = {}
class2prop = {}
prop2questions= {}
prop2constraints = {}
class_counts = {}
ace2seed = {}
ace2wiki = {}

with open(processing_sheets_path+"/processing_sheets/seed_list.csv") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data:  
        ace_class = row[0]  
        seed_classes = row[1]
        ace2seed[ace_class] = seed_classes.split("\n")

seed2ace = {seed:ace for ace, seeds in ace2seed.items() for seed in seeds}

def get_ace(wd_class):
    return seed2ace[wd_class]

with open(processing_sheets_path+"/processing_sheets/final_sheet.csv") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data:  
        seed_class = row[0]  
        seed_label = row[1]
        seed_transitive_count = row[2]
        wikidata_class_label = row[3]
        wikidata_class = row[4]
        wikidata_class_count = row[5]
        property = row[6]
        property_label = row[7]
        property_count = row[8]
        known_as = row[9]
        data_type = row[10]
        constraint_label_list = eval(row[11]) if row[11] else []
        constraint_class_list = eval(row[12]) if row[12] else []
        tail_counts = eval(row[13]) if row[13] else []
        # change from "['square_Q174782:2', 'archaeological site_Q839954:1', 'street_Q79007:1']" to {'square_Q174782': 2, 'archaeological site_Q839954': 1, 'street_Q79007': 1}
        if tail_counts != -1:
            tail_counts = {i.split(":")[0].split("_")[-1]:int(i.split(":")[-1]) for i in tail_counts}
        questions = eval(row[14]) if row[14] else []
        #tail_counts = eval(row[6])
        class_counts[wikidata_class] = wikidata_class_count
        if seed2ace[seed_class] not in ace2wiki:
            ace2wiki[seed2ace[seed_class]] = set()
        ace2wiki[seed2ace[seed_class]].update([wikidata_class])
        if wikidata_class not in class2prop:
            class2prop[wikidata_class ] = set()
        class2prop[wikidata_class].update([property])
        if wikidata_class  not in prop2questions:
            prop2questions[wikidata_class] = {}
            prop2constraints[wikidata_class] = {}
        prop2questions[wikidata_class][property] = questions
        if tail_counts != -1:
            prop2constraints[wikidata_class][property] = list(tail_counts.keys()) + constraint_class_list
        else:
            prop2constraints[wikidata_class][property] = constraint_class_list

def get_event_arguments(event):
    arguments = []
    for argument in event["roles"]:
        argument_type = argument[1]
        argument_text = argument[2]
        arguments.append((argument_type, argument_text))
    return arguments
    
def identify_correct_classes(candidate_arguments, candidates, context):
    """
    Input: dictionary of wikidataclass candidates and their properties
    Output: dictionary of chosen wikidataclasses and their properties

    The function identifies Ace Event types of candidate wiki classes and merges the candidates into lists representing a given event type.
    Then, for each list of candidates it selects candidates with the most amount of properties.
    From the selected candidates a wikidata class is then chosen based on the least amount of occurrences of the wikidata class in our csv
    """

    c = {}
    classes = []
    # summing up probability scores
    for candidate in candidate_arguments:
        for property in candidate_arguments[candidate]:
            if candidate_arguments[candidate][property]:
                if candidate not in c:
                    c[candidate] = 0
                c[candidate] += candidate_arguments[candidate][property][1]
    if not c.values():
        return False
    tmp1 = {}
    tmp2 = {}

    for key_pair in c:
        event_class = key_pair[0]
        event_type = key_pair[1]
        score = c[key_pair]
        if event_type not in tmp1:
            tmp1[event_type] = []
        tmp1[event_type].append(event_class)
        tmp2[event_class] = score
    
    for event_type in tmp1:
        best = (0, 0)
        for event_class in tmp1[event_type]:
            if not best or best[1]<tmp2[event_class]:
                best = (event_class, tmp2[event_class])
        classes.append((best, event_type))
    return classes

def identify_correct_classes(candidate_arguments, candidates, context):
    """
    Input: dictionary of wikidataclass candidates and their properties
    Output: dictionary of chosen wikidataclasses and their properties

    The function identifies Ace Event types of candidate wiki classes and merges the candidates into lists representing a given event type.
    Then, for each list of candidates it selects candidates with the most amount of properties.
    From the selected candidates a wikidata class is then chosen based on the least amount of occurrences of the wikidata class in our csv
    """

    c = {}
    classes = []
    # summing up probability scores
    for candidate in candidate_arguments:
        for property in candidate_arguments[candidate]:
            if candidate_arguments[candidate][property]:
                if candidate not in c:
                    c[candidate] = 0
                c[candidate] += candidate_arguments[candidate][property][1]
    if not c.values():
        return False
    tmp1 = {}
    tmp2 = {}

    for key_pair in c:
        event_class = key_pair[0]
        event_type = key_pair[1]
        score = c[key_pair]
        if event_type not in tmp1:
            tmp1[event_type] = []
        tmp1[event_type].append(event_class)
        tmp2[event_class] = score
    
    for event_type in tmp1:
        best = (0, 0)
        for event_class in tmp1[event_type]:
            if not best or best[1]<tmp2[event_class]:
                best = (event_class, tmp2[event_class])
        classes.append((best, event_type))
    return classes


def select_most_precise_time_expression(arguments):
    # check for tuples
    tpls = [argument for argument in arguments if isinstance(argument[0].texts, tuple)]
    # if a tuple exists, return the most probable longest one.
    if tpls:
        tpls = sorted(tpls, key=lambda x: x[1], reverse=True)
        best = sorted(tpls, key=lambda x: len(str(x[0].texts)), reverse=True)[0]
        return best
    # if a tuple doesn't exist, return the longest string that contains the most probable answer
    else:
        # get best answer by probability
        best = sorted(arguments, key=lambda x: x[1], reverse=True)[0]
        # sort by length of answer
        arguments = sorted(arguments, key=lambda x:len(str(x[0].texts)), reverse=True)
        # select first answer that containst the highest probability from arguments sorted by probability and length
        #print(best)
        #print(arguments)
        for argument in arguments:
            if str(best[0].texts) in str(argument[0].texts):
                return argument
        return best



def eliminate_named_events(events, links):
    # if an event trigger is a link, remove that event from list of events
    new_events = []
    for event in events:
        if isinstance(event,list) and event:
            event = event[0]
        elif not event:
            continue
        new_events.append(event)
        event_trigger = event["trigger"]
        for link in links:
            text = links[link][0]["anchor_text"]
            if event_trigger in text:
                new_events = new_events[:-1]
                break
    return new_events
        
def eliminate_unnamed_events(events, links):
    new_events = []
    for event in events:
        new_events.append(event)
        event_trigger = event["trigger"]
        for link in links:
            text = links[link][0]["anchor_text"]
            if event_trigger not in text:
                continue
            else:
                new_events.append(event)
    return new_events


def generate_questions(wd_classes_lists, completed):
    all_ordered_properties = []
    lengths_of_property_lists = []
    merged_lists_of_candidates = {}
    #first merge all the classes by type
    wd_classes_lists = [wd_classes_list for wd_classes_list in wd_classes_lists if wd_classes_list]
    for wd_classes, label in wd_classes_lists:
        if not isinstance(wd_classes, list):
            for candidate_pair in wd_classes_lists:
                candidate = candidate_pair[0]
                label = candidate_pair[1]
                if label not in merged_lists_of_candidates:
                    merged_lists_of_candidates[label] = []
                merged_lists_of_candidates[label].append(candidate)
        else:
            merged_lists_of_candidates[label] = wd_classes
    for label in merged_lists_of_candidates:
        merged_lists_of_candidates[label] = list(set(merged_lists_of_candidates[label]))
    # sort properties of classes of a event type by the number of class they occur in
    for label in merged_lists_of_candidates:
        property_lists = [list(class2prop[wd_class]) for wd_class in merged_lists_of_candidates[label]]
        property_counter = Counter()
        for property_list in property_lists:
            property_counter.update(property_list)
        # list of properties sorted by how many times they occurred 
        ordered_properties = sorted(property_counter, key=property_counter.get, reverse=True)
        lengths_of_property_lists.append(len(ordered_properties))
        all_ordered_properties.append(ordered_properties)

    questions = []
    constraints = {}
    indices = {}
    i = 0
    # get most occuring property of each ACE event type
    for ordered_properties in itertools.zip_longest(*all_ordered_properties):
        if len(ordered_properties) != len(merged_lists_of_candidates):
            while True:
                print("Something wrong in generate questions")
        for property, label in zip(ordered_properties, merged_lists_of_candidates):
            if property==None:
                continue
            # generate questions by wikidata class
            for wd_class in merged_lists_of_candidates[label]:
                if property not  in prop2questions[wd_class]:
                    continue
                # if the property has been seen before, and a question was asked to fill it for that particular class
                if property in completed and list(set(completed[property]) & set([wd_class])):
                    continue
                # if the property has been seen before, and a question was asked to fill it for any of the superclasses of that class
                #if property in completed and list(set(completed[property]) & set(all_classes_superclasses[wd_class])):
                    #continue
                if wd_class not in constraints:
                    constraints[wd_class] = {}

                if property not in constraints[wd_class]:
                    constraints[wd_class][property] = prop2constraints[wd_class][property]
                
                new_questions = prop2questions[wd_class][property]
                questions += new_questions
                indices[(i, i+len(new_questions))] = (wd_class, property, label)
                i = i + len(new_questions)
        if questions:
            break # comment to ask all questions instead of just the top ones
            

    return questions, indices, constraints


def get_initial_candidates(event_type):
    ev_type = event_type
    if  ev_type in ace2seed:
        candidates = [candidate for candidate in list(ace2seed[ev_type]) if candidate in class2prop]
        wd_classes = (candidates,ev_type)
        return wd_classes
    else:
        return None
    
    

def get_new_candidates(candidates, candidate_arguments, filter):
    # Returns subclasses of candidates + candidates in a list

    # if there exists a subclass in our list od candidates that has retrieved an argument, remove its superclasses from the list of candidates
    tmp=candidates[:]
    for candidate_pair1 in candidate_arguments:
        candidate_class = candidate_pair1[0]
        for super_class in all_classes_superclasses[candidate_class]:
            for candidate_pair2 in tmp:
                if super_class in candidate_pair2 and super_class in candidates:
                    # condition removal by basing it on whether the subclass has higher confidence than the superclass
                    for property in candidate_arguments[candidate_pair1]:
                        subclass_confidence = candidate_arguments[candidate_pair1][property][1]
                        superclass_confidence = candidate_arguments[candidate_pair2][property][1]
                        if subclass_confidence > superclass_confidence:
                            candidates.remove(candidate_pair2)
                            # eliminate candidate from further questions
                            filter.append(candidate_pair2)
                        else:
                            candidates.remove(candidate_pair1)
                            filter.append(candidate_pair1)
    new_candidates = []
    for candidate_pair in candidates:
        candidate = candidate_pair[0]
        event_type = candidate_pair[1]
        new_candidates = []
        # check if a candidate has subclasses, if not continue
        if candidate not in all_classes_subclasses:
            continue
        for subclass in all_classes_subclasses[candidate]:
            # ignore subclasses that don't have any approved properties
            if subclass not in class2prop:
                continue
            new_candidates.append((subclass,event_type))
    new_candidates = list(OrderedDict.fromkeys(new_candidates))
    return new_candidates, candidate_arguments, filter
    

def propagate_arguments(events):
    last_location = 0
    last_datetime = 0
    for event in events:
        location_filled = False
        time_filled = False
        for m, argument in enumerate(event.arguments):
            empty_indices = []
            if argument.type == "LOCATION":
                location_filled = True
                last_location = argument
            if argument.type == "DATETIME":
                time_filled = True
                last_datetime = argument
            if argument == '':
                empty_indices.append(m)
        if not location_filled and last_location:
            event.arguments[empty_indices[0]] = last_location
            empty_indices = empty_indices[1:]
        if not time_filled and last_datetime:
            event.arguments[empty_indices[0]] = last_location
    return events
