import sys, os
from nltk.tokenize import sent_tokenize, word_tokenize  
import os
from classes import * 
import ndjson
from format_new_dataset import format_dataset
from utils import * 
from collections import Counter,  OrderedDict
from SPARQLWrapper import SPARQLWrapper, JSON
import shelve
import time
import json
import datefinder
import shelve
import csv
from prepare_data import format_dataset

dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')
types_dbo = shelve.open('subevents/shelves/types_dbo')
types_wd = shelve.open('subevents/shelves/types_wd')
instance_types = shelve.open("subevents/shelves/instance_types")


with open("processing_sheets/event_subevents.json","r") as f:
    events = json.load(f)


def first_stage_eval_data():
    directory = "data/event_articles"
    dataset = {}
    dygiepp_data = []
    i = 0
    k = 0
    mistakes = 0
    event_wd_not_in_events = 0
    d = []
    all = 0
    e = time.time()
    context_window = 0
    for u, filename in enumerate(os.listdir(directory)):
        print("%d out of %d completed"%(u, len(os.listdir(directory))))
        with open(directory+filename, "r") as f:
            json_list = ndjson.load(f)
        for g,js in enumerate(json_list):
            all+=1
            Article_obj = Article(js)
            parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
            try:
                parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
            except (KeyError):
                mistakes+=1
                continue
            if parent_event_wd_id not in events:
                event_wd_not_in_events+=1

            for input_paragraph in Article_obj.input_paragraphs:
                last_sentence_subevents = []
                for sentence in input_paragraph.sentences:
                    all_sentence_link_types = [link["types"] for link in sentence.links]
                    all_sentence_link_types = [link for links in all_sentence_link_types for link in links]
                    for link in sentence.links:
                        if "Event" in all_sentence_link_types:
                            if "Event" in link["types"]:
                                context_window = 1
                                subevent_link = link["target"]
                                #except (UnicodeDecodeError, AttributeError):
                                try:
                                    subevent_wd_id = dbpedia_ids_to_wikidata_ids[subevent_link] 
                                except (KeyError):
                                    d.append(subevent_wd_id)
                                    continue
                                # if subevent doesnt have partof event
                                i+=1
                                if i%10000==0:
                                    print(i)
                                #if subevent_wd_id not in events[parent_event_wd_id]:
                                    #continue
                                last_sentence_subevents.append((subevent_wd_id, subevent_link))
                                if parent_event_wd_id not in dataset:
                                    dataset[parent_event_wd_id] = {}
                                if subevent_wd_id +"@" + subevent_link not in dataset[parent_event_wd_id]:
                                    dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link] = {"samples":[], "groundtruth":{}}
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"].append((link, sentence.text, sentence.links))
                        elif context_window:
                            context_window = 0
                            for subevent in last_sentence_subevents:
                                subevent_wd_id, subevent_link = subevent
                                link, text, links = dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1]
                                text += sentence.text
                                links += sentence.links
                                dataset[parent_event_wd_id][subevent_wd_id +"@" + subevent_link]["samples"][-1] = (link, text, links)
    with open('evaluation/linked_sub-events/eval_data_with_context_window.txt', 'w') as f:
        json.dump(dataset,f)
    
def stage_one(data_path="data/event_articles"):
    first_stage_eval_data(data_path)

    with open('evaluation/linked_sub-events/eval_data_with_context_window.txt', 'r') as f:
        dataset = json.load(f)
    print("First stage done")

    subevents_properties = {}
    with open("preprocessing_sheets/all_subevent_properties.txt", "r") as f:
        lines = f.readlines()[1:]
        for row in lines:
            row = row.rstrip().split("\t")
            event, property, value = row[0], row[1], row[2]
            if event not in subevents_properties:
                subevents_properties[event] = {}
            if property not in subevents_properties[event]:
                subevents_properties[event][property] = []
            subevents_properties[event][property].append(value)

    d = []
    zzz = 0
    for event_wd_id in dataset:
        for subevent in dataset[event_wd_id]:
            number_of_sentences = len(dataset[event_wd_id][subevent]["samples"]) 
            try:
                subevent_wd_id, subevent_link = subevent.split("@")
            except ValueError:
                print(subevent)
            if subevent_wd_id not in subevents_properties:
                d.append(subevent_wd_id)
                continue
            subevent_properties = subevents_properties[subevent_wd_id]
            dataset[event_wd_id][subevent]["groundtruth"] = subevent_properties
    print(len(d))

    with open('evaluation/linked_sub-events/eval.txt', 'w') as f:
        json.dump(dataset,f)


    with open('evaluation/linked_sub-events/eval.txt', 'r') as f:
        dataset = json.load(f)

    class2prop = {}
    with open("processing_sheets/final_sheet.csv") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data:  
            wikidata_class = row[4]
            property = row[6]
            if wikidata_class not in class2prop:
                class2prop[wikidata_class ] = set()
            class2prop[wikidata_class].update([property])

    new_dataset = {}
    d0 = []
    d1 = []
    d2 = []
    d3 = []
    d4 = []
    d5 = []
    d6 = []
    for num, event_wd_id in enumerate(dataset):
        print("%d out %d"%(num, len(dataset)))
        for subevent in dataset[event_wd_id]:
            d0.append(subevent)
            try:
                subevent_wd_id, subevent_link = subevent.split("@")
            except ValueError:
                continue
            if subevent_wd_id in instance_types:
                try:
                    subevent_types = instance_types[subevent_wd_id]
                except(KeyError):
                    d2.append(subevent_link)
            else:
                d1.append(subevent_wd_id)
                continue
            subevent_types = list(subevent_types)
            #print(subevent_types)
            groundtruth_properties = dataset[event_wd_id][subevent]["groundtruth"]
            tmp = subevent_types
            for i in tmp:
                if i=="Q1656682":
                    continue
                if i not in class2prop:
                    d3.append(i)
                    subevent_types.remove(i)
                    continue
            if not subevent_types:
                d4.append(subevent_link)
                continue
            if len(subevent_types) > 1 and "Q1656682" in subevent_types:
                subevent_types.remove("Q1656682")
                d5.append(subevent_link)
            else:
                subevent_properties  = set()
                for subevent_type in subevent_types:
                    if subevent_type == "Q1656682":
                        for cl in class2prop:
                            subevent_properties.update(class2prop[cl])
                    elif subevent_type in class2prop:
                        subevent_properties.update(class2prop[subevent_type])
                if not subevent_properties:
                    d1+=subevent_types
                groundtruth_properties = {k:v for k, v in groundtruth_properties.items() if k in subevent_properties}
                groundtruth_values = {wd:k for k,v in groundtruth_properties.items() for wd in v}
                completed = set()
                for sample in dataset[event_wd_id ][subevent]["samples"]:
                    new_groundtruth = {}
                    sample_subevent = sample[0]
                    sample_context = sample[1]
                    sample_links = sample[2]
                    times = datefinder.find_dates(sample_context)
                    times = [time_expression.isoformat() for time_expression in times]
                    for link in sample_links:
                        if link["target"] in dbpedia_ids_to_wikidata_ids:
                            link_wd_id = dbpedia_ids_to_wikidata_ids[link["target"]]
                        else:
                            continue
                        # if a sentence link has a wikidataid that occurs in grountruth property values, store it
                        if link_wd_id in groundtruth_values:
                            property = groundtruth_values[link_wd_id]
                            completed.update(property)
                            if property not in new_groundtruth:
                                new_groundtruth[property] = []
                            new_groundtruth[property].append(link_wd_id)
                    # once we have gone through the links, we want to look for literals: quantities and time expressions. 
                    for time_expression in times:
                        if time_expression in groundtruth_values:
                            property = groundtruth_values[time_expression]
                            completed.update(property)
                            if property not in new_groundtruth:
                                new_groundtruth[property] = []
                            new_groundtruth[property].append(time_expression)
                    for token in sample_context.split(" "):
                        if token.isdigit() and token in groundtruth_values:
                            property = groundtruth_values[token]
                            completed.update(property)
                            if property not in new_groundtruth:
                                new_groundtruth[property] = []
                            new_groundtruth[property].append(time_expression)
                    if event_wd_id not in new_dataset:
                        new_dataset[event_wd_id] = {}
                    if subevent not in new_dataset[event_wd_id]:
                        new_dataset[event_wd_id][subevent] = {"samples":[]}
                    if new_groundtruth:
                        new_sample = {"sample_event_link":sample_subevent, "sample_context":sample_context, "sample_all_links":sample_links, "sample_groundtruth": new_groundtruth, "ground_event_types": subevent_types}
                        new_dataset[event_wd_id][subevent]["samples"].append(new_sample)
                        d6.append(new_sample)
                    #if "groundtruth" not in new_dataset:
                    # del new_dataset[event_wd_id ][subevent]["samples"]["groundtruth"]
    #print(set(d1))
    print("missed because no properties or instance type not identified:")
    print(len(d1))
    print(len(set(d1)))
    print("@@@\n bugs:")
    #print(set(d2))
    print(len(d2))
    print(len(set(d2)))
    print("@@@\nsubevent_types removed because they have no properties:")
    #print(set(d3))
    print(len(d3))
    print(len(set(d3)))
    print("@@@\nsubevents removed because they have no types remaining:")
    #print(set(d4))
    print(len(d4))
    print(len(set(d4)))
    print("@@number of samples:")
    print(len(d6))
    print("@@ total subevents:")
    print(len(d0))






    with open('evaluation/linked_sub-events/raw.txt', 'w') as f:
        json.dump(new_dataset,f)

def stage_two():

    with open("evaluation/linked_sub-events/raw.txt","r") as f:
        ground = json.load(f)

    tmp = copy.deepcopy(ground)

    for event in tmp:
        for subevent in tmp[event]:
            if not tmp[event][subevent]["samples"]:
                del ground[event][subevent]

    tmp = copy.deepcopy(ground)           
    for event in tmp:
        if not ground[event]:
            del ground[event]

    new_ground = {}
    for event in ground:
        for subevent in ground[event]:
            for sample in ground[event][subevent]["samples"]:
                context = sample["sample_context"]
                if context not in new_ground:
                    new_ground[context] = []
                new_ground[context].append({"event_class":sample["ground_event_types"], "event_properties":sample["sample_groundtruth"]})

    with open("evaluation/linked_sub-events/ground.txt","w") as f:
        json.dump(new_ground, f)

    with open("evaluation/linked_sub-events/ground.txt","r") as f:
        new_ground = json.load(f)

    for i, context in enumerate(new_ground):
        os.mkdir("evaluation/linked_sub-events/unprocessed_data/"+str(i))
        with open("evaluation/linked_sub-events/unprocessed_data/"+str(i)+"/"+str(i)+".txt","w") as f:
            f.write(context)
        os.mkdir("evaluation/linked_sub-events/processed_data/"+str(i))
        format_dataset("evaluation/linked_sub-events/unprocessed_data/"+str(i), "evaluation/linked_sub-events/processed_data/"+str(i) ,"ace-event")



if __name__=="__main__":
    stage_one()
    stage_two()