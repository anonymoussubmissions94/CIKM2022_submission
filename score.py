import json
import csv 
from collections import Counter
import os
import copy 

ancestors = dict()

def establish_transitibity(d):
    hd = dict()
    for i, child in enumerate(d):
        if child not in hd:
            hd[child] = []
        hd[child] = get_parents(d, child, [])
        if i%100==0:
            print(i, len(d))
    return hd


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

with open("processing_sheets/all_event_superclasses.txt", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter='\t')
    for row in data:
        if row[0] not in ancestors:
            ancestors[row[0]] = []
        ancestors[row[0]].append(row[1]) 


for key in ancestors:
    if "Q35120" in ancestors[key]:
        ancestors[key].remove("Q35120")
    if "Q1190554" in ancestors[key]:
        ancestors[key].remove("Q1190554")
    if "Q1656682" in ancestors[key]:
        ancestors[key].remove("Q1656682")
        
ancestors = establish_transitibity(ancestors)

ancestors["Q17912683"] = ancestors["Q2380335"]
for key in ancestors:
    if "Q17912683" in ancestors[key]:
        ancestors[key].append("Q2380335")
        #ancestors[key].remove("Q17912683")

for key in ancestors:
    if key not in ancestors[key]:
        ancestors[key].append(key)

def return_closest_class(ground_classes, predictions):
    d = {}
    tmp = {}
    for prediction in predictions:
        predicted_event_class = prediction["predicted_event_class"]
        if predicted_event_class in ground_classes:
            ground_class = predicted_event_class
            return (ground_class, prediction)
        else:
            if predicted_event_class not in tmp:
                    tmp[predicted_event_class] =  prediction
            else:
                if len(tmp[predicted_event_class]["predicted_event_properties"]) < len(prediction["predicted_event_properties"]):
                    tmp[predicted_event_class] = prediction
            for ground_class in ground_classes:
                if (ground_class, predicted_event_class) not in d:
                    d[(ground_class, predicted_event_class)] = len(set(ancestors[ground_class]) & set(ancestors[predicted_event_class]))/len(ancestors[ground_class])
    d = dict(sorted(d.items(), key=lambda item: item[1]))
    ground_class, predicted_event_class = list(d.keys())[0]
    return ground_class, tmp[predicted_event_class]

def return_closest(ground_classes, predicted_class):
    max = 0
    for ground_class in ground_classes:
        if max == 0:
            max = len(set(ancestors[ground_class]) & set(ancestors[predicted_class]))/len(ancestors[ground_class])
            best = ground_class
        if len(set(ancestors[ground_class]) & set(ancestors[predicted_class]))/len(ancestors[ground_class]) > max:
            max = len(set(ancestors[ground_class]) & set(ancestors[predicted_class]))/len(ancestors[ground_class])
            best = ground_class,
    return ground_class


def hp(dataset):

    all_predicted_our_ver = 0
    all_predicted_t2e_ver = 0
    all_predicted_dygiepp_ver = 0

    all_ground_our_ver = 0
    all_ground_t2e_ver = 0
    all_ground_dygiepp_ver = 0

    our_plus = 0
    t2e_plus = 0
    dygiepp_plus = 0
    total_property = 0


    dygiepp_property_tp = 0
    our_property_tp = 0
    t2e_property_tp = 0

    total_dygiepp_classes = 0
    total_t2e_classes = 0
    total_our_classes = 0

    for context in dataset:
        ground = dataset[context]["ground"]
        for x in ground:
            ground_event_class = x["event_class"]
            ground_event_properties = x["event_properties"]
            
            dygiepp_predictions =  dataset[context]["dygiepp"]
            t2e_predictions = dataset[context]["t2e"]
            our_predictions = dataset[context]["our"]
            best = {}
            total_property += len(ground_event_properties)
            if dygiepp_predictions:
                for i, dygiepp_prediction in enumerate(dygiepp_predictions):
                    if not dygiepp_prediction:
                        continue
                    if not dygiepp_prediction["predicted_event_properties"]:
                        continue
                    total_dygiepp_classes += 1
                    if i not in best:
                        best[i] = 0
                    for property in ground_event_properties:
                        if property in dygiepp_prediction["predicted_event_properties"] and dygiepp_prediction["predicted_event_properties"][property] in ground_event_properties[property]:
                            best[i] += 1
                if best:
                    best_index = list({k:v for k,v in sorted(best.items(), key=lambda x:x[1], reverse=True)}.keys())[0]
                    dygiepp_prediction = dygiepp_predictions[best_index]
                    dygiepp_property_tp += best[best_index]
                    dygiepp_class = dygiepp_prediction["predicted_event_class"]
                    tmp = copy.deepcopy(ground_event_class)
                    if len(ground_event_class)>1:
                        tmp = return_closest(tmp, dygiepp_class)
                    else:
                        tmp = tmp[0]
                    all_ground_dygiepp_ver += len(set(ancestors[tmp]))
                    dygiepp_plus += len(set(ancestors[tmp]) & set(ancestors[dygiepp_class]))
                    all_predicted_dygiepp_ver += len(set(ancestors[dygiepp_class]))

            best = {}
            if t2e_predictions:
                for i, t2e_prediction in enumerate(t2e_predictions):
                    if not t2e_prediction["predicted_event_properties"]:
                        continue
                    total_t2e_classes += 1
                    if i not in best:
                        best[i] = 0
                    for property in ground_event_properties:
                        if property in t2e_prediction["predicted_event_properties"] and t2e_prediction["predicted_event_properties"][property] in ground_event_properties[property]:
                            best[i] += 1
                if best:
                    best_index = list({k:v for k,v in sorted(best.items(), key=lambda x:x[1], reverse=True)}.keys())[0]
                    t2e_prediction = t2e_predictions[best_index]
                    t2e_property_tp += best[best_index]
                    t2e_class = t2e_prediction["predicted_event_class"]
                    tmp = copy.deepcopy(ground_event_class)
                    if len(ground_event_class)>1:
                        tmp = return_closest(tmp, t2e_class)
                    else:
                        tmp = tmp[0]
                    all_ground_t2e_ver += len(set(ancestors[tmp]))
                    t2e_plus += len(set(ancestors[tmp]) & set(ancestors[t2e_class]))
                    all_predicted_t2e_ver += len(set(ancestors[t2e_class]))

            best = {}
            if our_predictions:
                for i, our_prediction in enumerate(our_predictions):
                    if not our_prediction["predicted_event_properties"]:
                        continue
                    total_our_classes += 1
                    if i not in best:
                        best[i] = 0
                    for property in ground_event_properties:
                        if property in our_prediction["predicted_event_properties"]: 
                            best[i] += len(set(our_prediction["predicted_event_properties"][property]) & set(ground_event_properties[property]))
                if best:
                    best_index = list({k:v for k,v in sorted(best.items(), key=lambda x:x[1], reverse=True)}.keys())[0]
                    our_prediction = our_predictions[best_index]
                    our_property_tp += best[best_index]
                    our_class = our_prediction["predicted_event_class"]
                    tmp = copy.deepcopy(ground_event_class)
                    if isinstance(our_class,list) and len(our_class)>1:
                        while True:
                            print("hey")
                    if isinstance(our_class, list):
                        our_class = our_class[0]
                    if len(ground_event_class)>1:
                        tmp = return_closest(tmp, our_class)
                    else:
                        tmp = tmp[0]
                    all_ground_our_ver += len(set(ancestors[tmp]))
                    our_plus += len(set(ancestors[tmp]) & set(ancestors[our_class]))
                    all_predicted_our_ver += len(set(ancestors[our_class]))
            
    our_hp = our_plus/all_predicted_our_ver
    our_hr = our_plus/all_ground_our_ver
    our_property_r = our_property_tp/total_property

    t2e_hp = t2e_plus/all_predicted_t2e_ver
    t2e_hr = t2e_plus/all_ground_t2e_ver
    t2e_property_r = t2e_property_tp/total_property

    dygiepp_hp = dygiepp_plus/all_predicted_dygiepp_ver
    dygiepp_hr = dygiepp_plus/all_ground_dygiepp_ver
    dygiepp_property_r = dygiepp_property_tp/total_property
    #property_r = property_plu15, dygies/all_ground_properties
    
    s1 = "%.2f" % total_t2e_classes, "&", "%.2f" % t2e_property_tp, "&", "%.2f" % total_dygiepp_classes, "&","%.2f" % dygiepp_property_tp, "&","%.2f" %  total_our_classes,"&","%.2f" % our_property_tp," \\"
    s2 = "Text2Event &","%.2f" % t2e_hp, "&", "%.2f" % t2e_hr, "&", "%.2f" % (2 * (t2e_hp * t2e_hr) / (t2e_hp + t2e_hr)), "&","%.2f" % t2e_property_r, " \\"
    s3 = "DyGIE++ &","%.2f" % dygiepp_hp, "&", "%.2f" % dygiepp_hr, "&", "%.2f" % (2 * (dygiepp_hp * dygiepp_hr) / (dygiepp_hp + dygiepp_hr)), "&", "%.2f" % dygiepp_property_r, " \\"
    s4 = "Our Approach &","%.2f" % our_hp, "&", "%.2f" % our_hr, "&", "%.2f" % (2 * (our_hp * our_hr) / (our_hp + our_hr)), "&","%.2f" % our_property_r, " \\\\ \\bottomrule"

    print(s1)
    print(s2)
    print(s3)
    print(s4)
    
    with open("evaluation/linked_sub-events/results/results.txt","w") as f:
        f.write(s1)
        f.write("\n")
        f.write(s2)
        f.write("\n")
        f.write(s3)
        f.write("\n")
        f.write(s4)
    