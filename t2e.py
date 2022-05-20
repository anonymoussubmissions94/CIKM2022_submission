#!/usr/bin/env python
# -*- coding:utf-8 -*-
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import configparser

from seq2seq.constrained_seq2seq import decode_tree_str
from extraction.predict_parser.tree_predict_parser import TreePredictParser
from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder

device = torch.device("cuda:0")
with torch.cuda.device('cuda:0'):
        torch.cuda.empty_cache()

class EventExtractor:
    def __init__(self, tokenizer, model, tree_parser, constraint_decoder):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.tree_parser = tree_parser
        self.constraint_decoder = constraint_decoder
    def from_pretrained(model_path):
        event_schema = EventSchema.read_from_file(f"{model_path}/event.schema")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        tree_parser = TreePredictParser(event_schema)
        constraint_decoder = get_constraint_decoder(tokenizer=tokenizer,
                                                    type_schema=event_schema,
                                                    decoding_schema="tree",
                                                    source_prefix='event: ')
        return EventExtractor(tokenizer=tokenizer,
                              model=model,
                              tree_parser=tree_parser,
                              constraint_decoder=constraint_decoder)
    def extract_event(self, text_list, constrained_decoding=False):
        text_list = ['event: ' + text for text in text_list]
        input_ids = self.tokenizer(text_list, return_tensors='pt',
                                   padding=True).input_ids
        input_ids = input_ids.to(device)
        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = input_ids[batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence, tgt_generated=sent)
        outputs = self.model.generate(
            input_ids,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if constrained_decoding else None
        )
        event = decode_tree_str(outputs.cpu(), self.tokenizer)
        event_list, _ = self.tree_parser.decode(pred_list=event, gold_list=[])
        event_list = [event['pred_record'] for event in event_list]
        return event_list

config = configparser.ConfigParser()
config.read("config.ini")
t2e_models = config.get("Paths","t2e_models")

model_path = t2e_models+"/"+"/dyiepp_ace2005_en_t5_base"
event_extractor = EventExtractor.from_pretrained(model_path=model_path)

def predict_events(texts):
    #with torch.cuda.device('cuda:1'):
        #torch.cuda.empty_cache()
    results = []
    events = event_extractor.extract_event(texts)
    for text, event in zip(texts, events):
        results.append((text, event))
    return results

def predict_sentence_events(sentences):
    return event_extractor.extract_event(sentences)
"""
if __name__ == "__main__":
    model_path = "pretrained_models/dyiepp_ace2005_en_t5_base"
    event_extractor = EventExtractor.from_pretrained(model_path=model_path)
    texts = ["Trump visits China.", "Trump's visit to Japan started in November and ended in December.", "Hello world.", "The man was captured in Los Angeles on Tuesday by bounty hunters."]
    #with open("data/myanmar.txt","r") as f:
        #texts = f.readlines()
    events = event_extractor.extract_event(texts)
    for text, event in zip(texts, events):
        print(text, event)
"""
#model_path = "pretrained_models/dyiepp_ace2005_en_t5_base"
#event_extractor = EventExtractor.from_pretrained(model_path=model_path)
