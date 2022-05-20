import csv
from doctest import ELLIPSIS_MARKER
import re
from dateparser import *
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy import number
from collections import Counter,  OrderedDict
import itertools
import spacy
nlp = spacy.blank("en")
nlp.add_pipe('opentapioca')


ddp = DateDataParser()
all_classes_subclasses = dict()
all_classes_superclasses = dict()


import pandas as pd
import csv
import shelve


dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')
types_dbo = shelve.open('subevents/shelves/types_dbo')
types_wd = shelve.open('subevents/shelves/types_wd')
types_shelve = shelve.open('subevents/shelves/instance_types')




def semantify(arguments):
    for argument in arguments:
        for link in argument.links:
            if link in dbpedia_ids_to_wikidata_ids:
                argument.wikidata_id.append(dbpedia_ids_to_wikidata_ids[link])
            if link in types_dbo:
                argument.dbpedia_types.append(types_dbo[link])
            if link in types_wd:
                argument.wikidata_types.append(list(types_wd[link]))

    return arguments
    

def assign_ner0(arguments, predicted_ner):
    #input: 
    # arguments: extracted argument objects
    # predicted_ner: predicted named entities in the span of the sentences from which the arguments where extracted
    for argument in arguments:
        for argument_text in argument.texts:
            for NE in predicted_ner:
                if NE in argument_text:
                    if NE in dbpedia_ids_to_wikidata_ids:
                        argument.wikidata_id.append(dbpedia_ids_to_wikidata_ids[NE])
                    if NE in types_dbo:
                        argument.dbpedia_types.append(types_dbo[NE])
                    if NE in types_wd:
                        argument.wikidata_types.append(list(types_wd[NE]))
    return arguments

import json 
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
    
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
processing_sheets_path = config.get("Paths","processing_sheets_path")


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



#sia = SentimentIntensityAnalyzer()
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

def applyArgumentConstraints(event):
    pass

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

def limit2(sentences):
    paragraphs = []
    for sentence in sentences:
        if paragraphs == []:
            paragraphs.append(sentence[:512])
        if len(paragraphs[-1] + " " + sentence) > 512:
            paragraphs.append(sentence)
        else:
            paragraphs[-1] = paragraphs[-1] + " " + sentence[:512]
    return paragraphs


def limit(input_text):
    texts = []
    input_text = [t.replace("  ", " ").replace(" . ", ". ") for t in input_text]
    if max([len(text) for text in input_text]) > 512: 
        for text in input_text:
            if len(text) <= 512:
                texts.append(text)
            else:
                sentences = sent_tokenize(text)
                temp = []
                for sentence in sentences:
                    if not temp:
                        temp.append(sentence)
                    elif len(temp[-1]+ " " + sentence) >512:
                        temp.append(sentence)
                    else:
                        temp[-1] = temp[-1]+ " " + sentence
                texts += temp 
        return texts
    return input_text

def get_event_arguments(event):
    arguments = []
    for argument in event["roles"]:
        argument_type = argument[1]
        argument_text = argument[2]
        arguments.append((argument_type, argument_text))
    return arguments
    
def substitute_with_trigger(event_type, event_trigger):
    questions = triggerDict[event_type]
    #questions = [question.replace("*", event_trigger) for question in questions]
    return questions

def old_identify_correct_class(properties, initial_candidates):
    """
    Input: (properties, class)
    Returns appropriate wikidata class for given properties answered by QA component
    """
    if not initial_candidates:
        return False
    tmp = initial_candidates
    for wd_class in tmp:
        for property in properties:
            if property not in list(class2prop[wd_class]):
                initial_candidates.remove(wd_class)
    numbers_of_instances = [class_counts[wd_class] for wd_class in initial_candidates]
    idx = min(enumerate(numbers_of_instances), key=lambda x: x[1])[0]
    wd_class = initial_candidates[idx]
    return wd_class

def identify_correct_class(candidate_arguments):
    """
    Input: (properties, class)
    Returns appropriate wikidata class for given properties answered by QA component
    """
    c = Counter()
    for candidate in candidate_arguments:
        for property in candidate_arguments[candidate]:
            if candidate_arguments[candidate][property]:
                c.update({candidate:1})
    if c.values():
        return c.most_common(1)[0]
    return False


def identify_correct_classes2(candidate_arguments, candidates, context):
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

def identify_correct_classes3(candidate_arguments):
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
    all_properties = []

    for event_type in tmp1:
        best = (0, 0)
        if event_type not in all_properties:
            all_properties[event_type] = []
        for event_class in tmp1[event_type]:
            all_properties[event_type].append(candidate_arguments[(event_type, event_class)])

    for event_type in all_properties:
        for i, properties in enumerate(all_properties[event_type]):
            for property in properties:
                for properties2 in all_properties[event_type][i+1:]:
                    properties[property] != properties2[property2]


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
    tmp = Counter()
    completed = {}
    classes = []
    # summing up confidence scores
    for candidate in candidate_arguments:
        for property in candidate_arguments[candidate]:
            if candidate_arguments[candidate][property]:
                if candidate not in c:
                    c[candidate] = 0
                c[candidate] += candidate_arguments[candidate][property][1]
    if not c.values() or not candidates:
        return False

    merged_lists_of_candidates = {}
    for candidate_pair in candidates:
        candidate = candidate_pair[0]
        label = candidate_pair[1]
        if label not in merged_lists_of_candidates:
            merged_lists_of_candidates[label] = []
        merged_lists_of_candidates[label].append(candidate)
    #merged_lists_of_candidates = [(candidates, label) for label, candidates in merged_lists_of_candidates.items()]

    for label in merged_lists_of_candidates:
        best = {}
        completed = {}
        for candidate in merged_lists_of_candidates[label]:
            if (candidate, label) in candidate_arguments:
                for property in candidate_arguments[(candidate, label)]:
                    if property not in completed:
                        completed[property] = []
                    completed[property].append(candidate)
        if completed == {}:
            continue
        property = sorted(completed.keys(), key=lambda x: len(completed[x]), reverse=True)[0]
        # create a dictionary of completed properties, where all but one property are completed 
        for prop in class2prop[candidate]:
            if prop not in completed:
                completed[prop] = []
            completed[prop].append(candidate)
        del completed[property]
        one_type_candidates = [(merged_lists_of_candidates[label], label)]
        questions, indices, constraints = generate_questions3(one_type_candidates, completed)
        qa_answers = run_QA(questions, context)
        for start,end in indices:
            candidate = indices[(start,end)][0]  
            property = indices[(start,end)][1]  
            event_type = indices[(start,end)][2]  
            if any(qa_answers[start:end]): 
                X = [answer for answer in qa_answers[start:end] if answer]
                best[candidate] =  sorted(X, key=lambda x: x[1], reverse=True)[0][1]
        #if best != {}:
        best_candidate = max(best, key=best.get)
        classes.append((best_candidate, label))
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
    

def get_questions2(event_type):
    """
    wd classes have properties those properties have coresponding questions
    this function accepts an ace event_type and returns:
        candidate wikidata_classes (all subclasses of corresponding seed wikidata classes): ["Q1","Q2",...,"Qn"]
        lists of their properties: [["P1","P2",..,"Pm"],["P1","P3",...],...,["P1",...]]     dimensions: [n*[...]]
        list of questions corresponding to properties: [["When did (...)", ....],...,["...?"]]  dimensions: [n*[...]]
        list of constraints for said properties: [[[<constraints of P1>], [<constraints of P2>],...[]], [[...],[...]]]


    return questions, properties, constraints
    """
    wd_classes = list(ace2wiki[event_type])
    property_lists = [list(class2prop[wd_class]) for wd_class in wd_classes]
    questions = []
    constraints_lists = []
    for property_list,  wd_class in zip(property_lists, wd_classes):
        tmpQ = []
        tmpC = []
        for property in property_list:
            tmpQ.append(prop2questions[wd_class][property])
            tmpC.append(prop2constraints[wd_class][property])
            # in case of multiple questions for one property
            # tmpP +=[property] * len(prop2questions[wd_class][property])
            # assumes all questions per property expect same type of object as an answer
            # constraints_lists += [constraints] * len(prop2questions[wd_class][property])
        questions.append(tmpQ)
        constraints_lists.append(tmpC)
    return questions, property_lists, constraints_lists, wd_classes

    

def limit_paragraph_size(large_paragraphs):
    #for large_paragraph in large_paragraphs:
    return large_paragraphs
         
def get_context_window(input_text, i, window = 0):
    # Create condition to not go limit window size if there are mulitple of the same event type
    if not window:
        return input_text[i]
    else:
        if i-window <= 0:
            text = " ".join(input_text[0:window+1])
            return text
        elif i == len(input_text)-1:
            text =  " ".join(input_text[i-window:]) 
            return text
        else:
            text = " ".join(input_text[i-window: i+window+1])
            return text


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


def assign_links(arguments, links, paragraph):
    #links are indexed on a sentence basis, not the paragraph 
    for argument in arguments:
        # straightforward case: argument text == link text
        if argument.text in links:
            argument.link = links[argument.text][0]["target"]
            links[argument.text] = links[argument.text][1:]
            if links[argument.text] == []:
                links.pop(argument.text, None)
            continue
        # argument text has an article, but otherwise identical to link text
        elif "the " in argument.text or "The " in argument.text:
            temp = argument.text.replace("The ", "")
            temp = argument.text.replace("the ", "")
            if temp in links:
                argument.link = links[temp][0]["target"]
                links[temp] = links[temp][1:]
                if links[temp] == []:
                    links.pop(temp, None)
                continue
        s,e  = re.search(re.escape(argument.text), paragraph).span()
        for key in links:
            # if the argument is contained within the link
            if links[key][0]["start"] <= s and links[key][0]["end"] >= e:
                argument.link = links[key][0]["target"]
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                    break
            # if the link is contained within the argument
            elif s <= links[key][0]["start"] and  e >= links[key][0]["end"]:
                argument.link = links[key][0]["target"]
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                    break
        """
        elif False:
            for link in links:
                if link in argument.text:
                    print("argument text: " + argument.text)
                    print("link: "+str(link))
                    print()
                    print(links[link])
                    argument.link = links[link][0]["target"]
                    links[link] = links[link][1:]
        """
    return arguments, links

def old_assign_links2(arguments, links, paragraph):
    for argument in arguments:
        print(paragraph)
        print("--------------")
        print(argument)
        s,e  = re.search(re.escape(argument.text), paragraph).span()
        tmp_links = dict(links)
        for key in tmp_links:
            # if the argument is contained within the link
            if links[key][0]["start"] <= s and links[key][0]["end"] >= e:
                argument.links = links[key][0]["target"]
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                #break uncomment for one link per argument
            # if the link is contained within the argument
            elif s <= links[key][0]["start"] and  e >= links[key][0]["end"]:
                argument.links = links[key][0]["target"]
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                #break uncomment for one link per argument
    return arguments, links

def assign_links3(arguments, links, paragraph):
    matches = []
    for argument in arguments:
        #print(paragraph)
        #print("--------------")
        #print(argument)
        for sentence in sent_tokenize(paragraph):
            sentence_matches = [(m.start(), m.end()) for m in re.finditer(re.escape(argument.text), sentence)]
            if sentence_matches:
                matches += sentence_matches
        #print(links)
        tmp_links = dict(links)
        for key in tmp_links:
            # if the link is contained within the argument: start of the argument is before/including start of link up to after/including end of link
            # eg: argument = "President of US Barack Obama", link = "Barack Obama"
            # example with multiple links, argument = "Germany, Russia and Ukraine", link = "Germany", link = "Russia", link = "Ukraine"
            for match in matches:
                s = match[0]
                e =  match[1]
                if s <= links[key][0]["start"] and  e >= links[key][0]["end"]:
                    argument.links.append(links[key][0]["target"])
                #links[key] = links[key][1:]
                #if links[key] == []:
                    #links.pop(key, None)
                #break uncomment for one link per argument

    return arguments, links



def assign_links4(arguments, links, paragraph):
    matches = []
    for argument in arguments:
        for text in argument.texts:
            for sentence in sent_tokenize(paragraph):
                sentence_matches = [(m.start(), m.end()) for m in re.finditer(re.escape(text), sentence)]
                if sentence_matches:
                    matches += sentence_matches
            tmp_links = dict(links)
            for key in tmp_links:
                for match in matches:
                    s = match[0]
                    e =  match[1]
                    if s <= links[key][0]["start"] and  e >= links[key][0]["end"]:
                        argument.links.append(links[key][0]["target"])
                        argument.links = list(set(argument.links))

    return arguments, links

def assign_links2(arguments, links, paragraph):
    for argument in arguments:
        print(paragraph)
        print("--------------")
        print(argument)
        s,e  = re.search(re.escape(argument.text), paragraph).span()
        tmp_links = dict(links)
        for key in tmp_links:
            # if the argument is contained within the link: start of argument text is after/including link up to or less than the full link,
            #  eg: argument = "Obama", link = "Barack Obama"
            if links[key][0]["start"] <= s and links[key][0]["end"] >= e:
                argument.links.append(links[key][0]["target"])
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                #break uncomment for one link per argument
            # if the link is contained within the argument: start of the argument is before/including start of link up to after/including end of link
            # eg: argument = "President of US Barack Obama", link = "Barack Obama"
            # example with multiple links, argument = "Germany, Russia and Ukraine", link = "Germany", link = "Russia", link = "Ukraine"
            elif s <= links[key][0]["start"] and  e >= links[key][0]["end"]:
                argument.links.append(links[key][0]["target"])
                links[key] = links[key][1:]
                if links[key] == []:
                    links.pop(key, None)
                #break uncomment for one link per argument
    return arguments, links

def produce_context_links(context, input_paragraphs):
    temp = 0
    all_links = []
    links = {}
    stepped_into_range = False
    for paragraph in input_paragraphs:
        for sentence in paragraph.sentences:
            pattern = sentence.text
            if bool(re.search(re.escape(pattern), context)):
                stepped_into_range = True
                if sentence.links:
                    all_links += [link for link in sentence.links]
                    for i in range(len([link for link in sentence.links])):
                        all_links[-len([link for link in sentence.links])+i]["start"] += temp
                        all_links[-len([link for link in sentence.links])+i]["end"] += temp
                temp += len(sentence.text) + 1  #the space between . and the next word
            elif stepped_into_range:
                for link in all_links:
                    if link["anchor_text"] not in links:
                        links[link["anchor_text"]] = []
                    links[link["anchor_text"]].append(link)
                return links

def produce_context_links2(context, input_paragraphs):
    all_links = []
    links = {}
    original_sentence = ''
    if isinstance(context, list):
        temp = 0
        for paragraph in input_paragraphs:
            for sentence in paragraph.sentences:
                pattern = word_tokenize(sentence.text)
                if  pattern == context:
                    original_sentence = sentence.text
                    if sentence.links:
                        all_links += [link for link in sentence.links]
                        for i in range(len([link for link in sentence.links])):
                            all_links[-len([link for link in sentence.links])+i]["start"] += temp
                            all_links[-len([link for link in sentence.links])+i]["end"] += temp
                    temp += len(sentence.text) + 1  
                    
    else:              
        for context_sentence in sent_tokenize(context):
            temp = 0
            for paragraph in input_paragraphs:
                for sentence in paragraph.sentences:
                    pattern = sentence.text
                    if bool(re.search(re.escape(pattern), context_sentence)):
                        original_sentence = sentence.text
                        if sentence.links:
                            all_links += [link for link in sentence.links]
                            for i in range(len([link for link in sentence.links])):
                                all_links[-len([link for link in sentence.links])+i]["start"] += temp
                                all_links[-len([link for link in sentence.links])+i]["end"] += temp
                        temp += len(sentence.text) + 1  #the space between . and the next word
    for link in all_links:
        if link["anchor_text"] not in links:
            links[link["anchor_text"]] = []
        links[link["anchor_text"]].append(link)
    return links, original_sentence





