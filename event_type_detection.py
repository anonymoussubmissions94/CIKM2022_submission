
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

def generate_unprocessed_data(i, data_path):
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
        os.mkdir("unprocessed_eval_data/"+str(i)+"/"+str(g))
        with open("unprocessed_eval_data/"+str(i)+"/"+str(g)+"/"+str(g)+".txt","a") as f2:
            for sentence in sentences:
                f2.write(sentence+" ")



if __name__ == "__main__":
    directory = project_path+"/data/new/"
    for i, file in enumerate(sorted(os.listdir(directory))):
        print("File: %s done. \n %d of %d completed."%(file, i, len(sorted(os.listdir(directory)))))
        os.mkdir("unprocessed_eval_data2/"+str(i)+"/")
        generate_unprocessed_data(i, directory+file)

# PROCESS DATA WITH process_eval_data.sh
# PREDICT with run_event_predictions
# Rerun predictions by running re-run_missed_prediction.py followed by rerun_event_predictions.sh
# Run merge_predictions
# Run reformat_predictions_to_wd.py



