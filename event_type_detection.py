
import ndjson
import sys, os
import re
import os
from utils import *
from nltk.tokenize import sent_tokenize, word_tokenize  
from numpy import number
import numpy as np
import json
from classes import * 
from random import shuffle
from collections import Counter,  OrderedDict
import shelve
import ndjson


config = configparser.ConfigParser()
config.read("config.ini")
project_path = config.get("Paths","project_path")

dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')

total_articles = 0
dataset = {}
all_sentences = []

def split(sentences, n):
    n = max(1, n)
    return [sentences[i:i+n] for i in range(0, len(sentences), n)]

def format_document(fname, dataset_name, nlp):
    text = open(fname).read()
    doc = nlp(text)
    sentences = [[tok.text for tok in sent] for sent in doc.sents]
    doc_key = os.path.basename(fname).replace(".txt", "")
    res = {"doc_key": doc_key,
           "dataset": dataset_name,
           "sentences": sentences}
    return res


def format_dataset(data_directory, output_file, dataset_name, use_scispacy):
    nlp_name = "en_core_sci_sm" if use_scispacy else "en_core_web_sm"
    nlp = spacy.load(nlp_name)

    fnames = [f"{data_directory}/{name}" for name in os.listdir(data_directory)]
    res = [format_document(fname, dataset_name, nlp) for fname in fnames]
    with open(output_file, "w") as f:
        for doc in res:
            print(json.dumps(doc), file=f)

def generate_unprocessed_data(i, data_path, project_path):
    f =  open(data_path, "r")
    total_articles = 0
    json_list = ndjson.load(f)  
    for g,js in enumerate(json_list):
        print("Article: " + str(total_articles)) 
        total_articles+=1
        Article_obj = Article(js)
        parent_event_link = ("_".join(Article_obj.article_name.split(" ")))
        try:
            parent_event_wd_id = dbpedia_ids_to_wikidata_ids[parent_event_link]
        except (KeyError):
            continue
        dataset[parent_event_wd_id] = {}
        input_paragraphs = Article_obj.input_paragraphs
        input_text = [" ".join(input_paragraph.text) for input_paragraph in input_paragraphs]   
        sentences = []
        # Skip articles with less than 5 sentences
        if len([sentence for sentence in input_text])<5: 
            continue
        for few_sentences in input_text:
            tokenized = sent_tokenize(few_sentences)
            sentences += tokenized
        sentences = [sentence for sentence in sentences if len(word_tokenize(sentence)) > 5]
        sentences = sent_tokenize(" ".join(sentences))
        sentences = [sentence +"."  if sentence[-1]!="." else sentence for sentence in sentences]
        os.mkdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/"+str(g))
        partitioned_sentences = split(sentences, 1000)
        for l, partition in enumerate(partitioned_sentences):
            with open("unprocessed_data/"+str(i)+"/"+str(g)+"/"+str(g)+"_"+str(l)+".txt","a") as f2:
                for sentence in sentences:
                    f2.write(sentence+" ")


def merge():
    import os
    import json
    directory = "predictions2/"
    started_file = False
    for subdirectory in sorted(os.listdir(directory)):
        for file in sorted(os.listdir(directory+subdirectory)):
            if not started_file:
                file_core = file.split("_")[0]
                predictions = {}
            if not(file.endswith("_1.txt") or file.endswith("_2.txt") or file.endswith("_3.txt") or file.endswith("_4.txt")):
                continue
            with open(directory+subdirectory+"/"+file,"r") as f:
                quarter_predictions = json.load(f)
            if file.endswith("_1.txt"):
                started_file = True
            predictions["doc_key"] = quarter_predictions["doc_key"]
            predictions["dataset"] =  quarter_predictions["dataset"]
            if "sentences" not in predictions:
                predictions["sentences"] = quarter_predictions["sentences"]
                if "predicted_ner" in quarter_predictions:
                    predictions["predicted_ner"] = quarter_predictions["predicted_ner"]
                if "predicted_relations" in quarter_predictions:
                    predictions["predicted_relations"] = quarter_predictions["predicted_relations"]
                if "predicted_events" in quarter_predictions:
                    predictions["predicted_events"] = quarter_predictions["predicted_events"]

            else:
                predictions["sentences"]+= quarter_predictions["sentences"]
                if "predicted_ner" in quarter_predictions:
                    if "predicted_ner" not in predictions:
                        predictions["predicted_ner"] = []
                    predictions["predicted_ner"]+= quarter_predictions["predicted_ner"]
                if "predicted_relations" in quarter_predictions:
                    if "predicted_relations" not in predictions:
                        predictions["predicted_relations"] = []
                    predictions["predicted_relations"]+= quarter_predictions["predicted_relations"]
                if "predicted_events" in quarter_predictions:
                    if "predicted_events" not in predictions:
                        predictions["predicted_events"] = []
                    predictions["predicted_events"]+= quarter_predictions["predicted_events"]
            if file.endswith("_4.txt"):
                started_file = False
                with open("predictions/"+file.split("_")[0]+".txt", "w") as f2:
                    json.dump(predictions, f2)


if __name__ == "__main__":
    print("Preparing unprocessed data")
    directory = project_path+"/data/event_articles/"
    for i, file in enumerate(sorted(os.listdir(directory))):
        print("File: %s done. \n %d of %d completed."%(file, i, len(sorted(os.listdir(directory)))))
        os.mkdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/")
        generate_unprocessed_data(i, directory+file)
    print("Preparing processed data")
    for folder_input_path in os.listdir(project_path+"/data/intermediate_results/unprocessed_data/"):
        output_path_sufix = folder_input_path.split("/")[-1]
        output_folder_path = project_path + "/data/intermediate_results/processed_data/" + output_path_sufix
        os.mkdir(output_folder_path)
        output_file_path  = project_path + "/data/intermediate_results/processed_data/" + output_path_sufix +"/" +  output_path_sufix +".txt"
        format_dataset(folder_input_path, output_file_path ,"ace-event")
    print("Ready for event type prediction")
    #print("Merging split file prediction")
    #merge_predictions()


    


# PROCESS DATA WITH process_eval_data.sh
# PREDICT with run_event_predictions
# Rerun predictions by running re-run_missed_prediction.py followed by rerun_event_predictions.sh
# Run merge_predictions
# Run reformat_predictions_to_wd.py



