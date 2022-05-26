
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
config.read("configs/config.ini")
project_path = config.get("Paths","project_path")

dbpedia_ids_to_wikidata_ids = shelve.open('subevents/shelves/dbpedia_to_wikidata_en')

total_articles = 0
dataset = {}
all_sentences = []

def split(sentences, n):
    n = max(1, n)
    return [sentences[i:i+n] for i in range(0, len(sentences), n)]




def split_processed_data(file_path, res):
    for i, doc in enumerate(res):
        doc_key = doc["doc_key"]
        dataset = doc["dataset"]
        sentence_partition = doc["sentences"][i*(len(doc["sentences"])//len(res)):(i+1)*(len(doc["sentences"])//len(res))]
        with open(file_path.split(".")[0]+"_"+str(i)+".txt", "w") as f:
            tmp = {"doc_key":doc_key, "dataset":dataset, "sentences":sentence_partition}
            json.dump(tmp, f)


def format_document(fname, dataset_name, nlp):
    text = open(fname).read()
    doc = nlp(text)
    sentences = [[tok.text for tok in sent] for sent in doc.sents]
    doc_key = os.path.basename(fname).replace(".txt", "")
    res = {"doc_key": doc_key,
           "dataset": dataset_name,
           "sentences": sentences}
    return res


def format_dataset(data_directory, output_file, dataset_name, use_scispacy=False):
    nlp_name = "en_core_sci_sm" if use_scispacy else "en_core_web_sm"
    nlp = spacy.load(nlp_name)

    fnames = [f"{data_directory}/{name}" for name in os.listdir(data_directory)]
    res = [format_document(fname, dataset_name, nlp) for fname in fnames]
    if len(res)==1:
        with open(output_file, "w") as f:
            for doc in res:
                print(json.dumps(doc), file=f)
    else:
        split_processed_data(output_file, res)

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
        if not os.path.isdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/"+str(g)):
            os.mkdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/"+str(g))
        partitioned_sentences = split(sentences, 100)
        for l, partition in enumerate(partitioned_sentences):
            with open("data/intermediate_results/unprocessed_data/"+str(i)+"/"+str(g)+"/"+str(g)+"_"+str(l)+".txt","a") as f2:
                for sentence in sentences:
                    f2.write(sentence+" ")





if __name__ == "__main__":
    print("Preparing unprocessed data")
    directory = project_path+"/data/event_articles/"
    if not os.path.isdir(project_path+"/data/intermediate_results/"):
        os.mkdir(project_path+"/data/intermediate_results/")
    if not os.path.isdir(project_path+"/data/intermediate_results/unprocessed_data"):
        os.mkdir(project_path+"/data/intermediate_results/unprocessed_data")
    for i, file in enumerate(os.listdir(directory)):
        print("File: %s done. \n %d of %d completed."%(file, i, len(os.listdir(directory))))
        if not os.path.isdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/"):
            os.mkdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(i)+"/")
        generate_unprocessed_data(i, directory+file, project_path)
    print("Preparing processed data")
    if not os.path.isdir("data/intermediate_results/processed_data/"):
        os.mkdir("data/intermediate_results/processed_data/")
    for n, input_folder in enumerate(sorted(os.listdir(project_path+"/data/intermediate_results/unprocessed_data/"), key=lambda z: int(z))):
        #output_path_sufix = input_folder.split("/")[-1]
        output_folder_path = project_path + "/data/intermediate_results/processed_data/" + str(n)
        if not os.path.isdir(output_folder_path):
            os.mkdir(output_folder_path)
        for subfolder in sorted(os.listdir(project_path+"/data/intermediate_results/unprocessed_data/"+str(n)),key=lambda z: int(z)):
            output_file_path  = project_path + "/data/intermediate_results/processed_data/" + str(n) +"/" +  str(subfolder) +".txt"
            format_dataset(project_path+"/data/intermediate_results/unprocessed_data/"+str(n) + "/" + subfolder, output_file_path ,"ace-event")
    print("Ready for event type prediction")
    
    #print("Merging split file prediction")
    #merge_predictions()

"""
    for directory in /home/kuculo/ExtractMore/text2event/evaluation/unproc/*; do
        /home/kuculo/anaconda3/envs/dygiepp/bin/python scripts/new-dataset/format_new_dataset.py $directory /home/kuculo/ExtractMore/text2event/evaluation/proc/${directory##*/}/${directory##*/}.txt ace-event  
    done
done
"""


# PROCESS DATA WITH process_eval_data.sh
# PREDICT with run_event_predictions
# Rerun predictions by running re-run_missed_prediction.py followed by rerun_event_predictions.sh
# Run merge_predictions
# Run reformat_predictions_to_wd.py



