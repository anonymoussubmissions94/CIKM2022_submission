import csv
from collections import Counter
from tabnanny import check
from turtle import update
seed2ace = dict()
parent_classes = dict()
# key is class, value is parent class
class_statistics = dict()
class_labels = dict()
property_labels = dict()
class_property_count = dict()
class_property_data_type = dict()
class_property_constraint = dict()
class2ace = dict()
filtered_out = []
super_sub_classes = {}
seeds_names = dict()
all_classes_subclasses = {}
all_classes_transitive_counts = {}
properties_per_class = {}
old_counts = {}
import configparser
config = configparser.ConfigParser()
config.read("config.ini")
processing_sheets_path = config.get("Paths","processing_sheets_path")

with open(processing_sheets_path+"/seed_list.csv", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data: 
        ace_label = row[0]  
        wd_classes = row[1].split("\n")
        for wd_class in wd_classes:
            seed2ace[wd_class] = ace_label


with open(processing_sheets_path+"/ace_subclasses.csv", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter='\t')
    for row in data: 
        if row[0] not in super_sub_classes:
            super_sub_classes[row[0]] = []
        super_sub_classes[row[0]].append(row[1]) 

with open(processing_sheets_path+"/all_classes_subclasses.csv", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        if row[0] not in all_classes_subclasses:
            all_classes_subclasses[row[0]] = []
        all_classes_subclasses[row[0]].append(row[1]) 


with open(processing_sheets_path+"/all_types_transitive_counts.csv", "r") as csvfile:
    next(csvfile)
    data = csv.reader(csvfile, delimiter=',')
    for row in data:
        all_classes_transitive_counts[row[0]] = row[1]

def create_sheet():
    wiki_type_seed = dict()

    D = []
    # getting seed transitive counts, and seed labels
    with open(processing_sheets_path+"/seeds_information.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data: 
            seeds_names[row[0]] = row[1]
            old_counts[row[0]] = row[2]


    # narrowing down seed classes for the sub classes
    with open(processing_sheets_path+"/seeds_subclasses.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data: 
            # choosing the more specific seed class in cases where there are multiple
            subclass = row[1]
            if subclass not in wiki_type_seed:
                wiki_type_seed[subclass] = []
            wiki_type_seed[subclass].append(row[0])
        for subclass in wiki_type_seed:
            seeds = wiki_type_seed[subclass]
            tmp = seeds[:]
            # foreach candidate seed
            for seed in tmp:
                if seed not in seed2ace:
                    seeds.pop(seeds.index(seed))
                    continue
                # if the current seed is a superclass of another seed
                if seed in super_sub_classes and len(seeds)>1:
                    # iterate through all subclasses of the current seed
                    for sub_class in super_sub_classes[seed]:
                        # if there is a subclass of the current seed in the list of candidate seeds, eject the current seed from list of candidates
                        if sub_class in seeds:
                            index = seeds.index(seed)
                            wiki_type_seed[subclass].pop(index)
                            #seeds.pop(index)
                            break
    # summing up subclass properties
    wd_labels = {}
    with open(processing_sheets_path+"/all.csv", "r") as csvfile:
        data = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(data): 
            if i==0:
                column_names = row
                continue
            wikidata_class_label = row[0]
            if wikidata_class_label == "FALSE":
                continue 
            wikidata_class = row[1]
            seed_classes = wiki_type_seed[wikidata_class]
            wd_labels[wikidata_class] = wikidata_class_label
            wikidata_class_count = int(row[2])
            transitive_wikidata_class_count = int(all_classes_transitive_counts[wikidata_class])
            prop = row[3]
            property_label = row[4]
            property_count = int(row[5])
            known_as = row[6]
            data_type = row[7]
            constraint_label_list = eval(row[8]) if row[8] else []
            constraint_class_list = eval(row[9]) if row[9] else []
            tail_counts = eval(row[10]) if row[10] else []

            if wikidata_class not in properties_per_class:
                properties_per_class[wikidata_class] = {}
            if prop not in properties_per_class[wikidata_class]:
                properties_per_class[wikidata_class][prop] = [property_label, property_count, known_as, data_type, constraint_label_list, constraint_class_list, tail_counts]#property_count

    print(len(properties_per_class))
    print(len(all_classes_subclasses))

    updated_properties_per_class = {}
    for wikidata_class in properties_per_class:
        proprow = {}
        if wikidata_class not in updated_properties_per_class:
            updated_properties_per_class[wikidata_class] = {}
    # go through all subclasses of a class

        for subclass in all_classes_subclasses[wikidata_class]:
            if subclass not in properties_per_class:
                #with open("missing_classes.txt", "r") as f:
                    #lineset = set(f)
                #with open("missing_classes.txt", "a") as f:
                    #if subclass+"\n" not in lineset:
                        #f.write(subclass+"\n")
                continue
            # avoiding double counting for seed classses
            if wikidata_class == subclass:
                continue
            # go through all properties of a subclass and sum up the counts, the constraints and the tails
            for prop in properties_per_class[subclass]:
                if prop not in proprow:
                    proprow[prop] = properties_per_class[subclass][prop][:]
                else:
                    proprow[prop][1] = int(proprow[prop][1])
                    proprow[prop][1] += int(properties_per_class[subclass][prop][1])


                    proprow[prop][4] += properties_per_class[subclass][prop][4] 

                    proprow[prop][5] += properties_per_class[subclass][prop][5]

                    proprow[prop][6] += properties_per_class[subclass][prop][6] 

                    proprow[prop][4] = list(set(proprow[prop][4]))
                    proprow[prop][5] = list(set(proprow[prop][5]))
                    proprow[prop][6] = list(set(proprow[prop][6]))

        for prop in properties_per_class[wikidata_class]:
            if prop not in updated_properties_per_class[wikidata_class]:
                updated_properties_per_class[wikidata_class][prop] = properties_per_class[wikidata_class][prop][:]
            # if property of a wikidata_class exists in its subclasses add up its count to that of its subclasses
            if prop in proprow:
                updated_properties_per_class[wikidata_class][prop][1] +=  proprow[prop][1]
                updated_properties_per_class[wikidata_class][prop][4] += proprow[prop][4]
                updated_properties_per_class[wikidata_class][prop][5] += proprow[prop][5]
                updated_properties_per_class[wikidata_class][prop][6] += proprow[prop][6]

    # writing down update full sheet
    for wikidata_class in updated_properties_per_class:
        for seed_class in wiki_type_seed[wikidata_class]:
            if seed_class in ["Q102252225", "Q685744", "Q3237993"]:
                continue
            else:
                seed_transitive_count = int(all_classes_transitive_counts[seed_class])
            wikidata_class_label = wd_labels[wikidata_class]
            transitive_wikidata_class_count = int(all_classes_transitive_counts[wikidata_class])
            seed_label = seeds_names[seed_class]
            for property in updated_properties_per_class[wikidata_class]:
                row = updated_properties_per_class[wikidata_class][property]
                prop = property
                property_label = row[0]
                transitive_property_count = row[1]
                known_as = row[2]
                data_type = row[3]
                constraint_label_list = row[4]
                constraint_class_list = row[5]
                tail_counts = row[6]
                D.append({"seed class": seed_class, "seed label": seed_label, "transitive_cnt":seed_transitive_count, "type_label": wikidata_class_label, "type": wikidata_class, "transitive_class_instances_cnt": transitive_wikidata_class_count, "property": prop,\
            "property_label": property_label, "count": transitive_property_count, "also_known":known_as,"data_type":data_type,"constraint_label_list":constraint_label_list, "constraint_class_list":constraint_class_list,"tail_cnt":tail_counts})
            
    with open(processing_sheets_path+"/data.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["seed class", "seed label", "transitive_cnt", "type_label", "type", "transitive_class_instances_cnt", "property","property_label",\
                "count","also_known","data_type","constraint_label_list", "constraint_class_list","tail_cnt"])
        writer.writeheader()
        for row in D:
            writer.writerow(row)


def reduce_number_of_properties(path = processing_sheets_path+"/data.csv", path2 = processing_sheets_path+"/reduced_data.csv"):
    white_list = ["http://wikiba.se/ontology#WikibaseItem","http://wikiba.se/ontology#Quantity","http://wikiba.se/ontology#Time"]
    class_black_list = []
    time_properties = ["P585","P2047","P571","P580","P582"]
    location_properties = ["P17","P276","P131"]
    base_properties = time_properties + location_properties + ["P710"] + ["P1132"]

    prop_labels = {"P585":"point in time","P2047":"duration","P571":"inception","P580":"start time","P582":"end time",\
        "P17":"country","P276":"location","P131":"located in the administrative territorial entity","P710":"participant", "P1132":"number of participants"}

    check_for_base_properties = {}
    set_of_accepted_classes = set()
    D = []

    property_black_list = []
    with open(processing_sheets_path+"/categories.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:          
            property_black_list.append(row[0].replace("http://www.wikidata.org/entity/",""))

    with open(processing_sheets_path+"/properties_to_remove.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:          
            property_black_list.append(row[0])

    with open(processing_sheets_path+"/classes_to_remove.csv","r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:
            if row[2]:         
                class_black_list.append(row[0])

    with open(path, "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=',')
        for row in data: 
            seed_class = row[0]  
            seed_label = row[1]
            seed_transitive_count = row[2]
            wikidata_class_label = row[3]
            wikidata_class = row[4]
            transitive_wikidata_class_count = row[5]
            prop = row[6]
            property_label = row[7]
            transitive_property_count = row[8]
            known_as = row[9]
            data_type = row[10]
            constraint_label_list = row[11]
            constraint_class_list = row[12]
            tail_counts = row[13]


            # add some base constraints to location properties
            if prop in location_properties:
                constraint_class_list = eval(constraint_class_list) + ["Q56061", "Q515", "Q6256", "Q2221906","Q17334923"]
                constraint_class_list = list(set(constraint_class_list))
                constraint_label_list = eval(constraint_label_list) + ["administrative territorial entity","city","country","geographic location","location"]
                constraint_label_list = list(set(constraint_label_list))

            # for each wikidata_class add its class information to a dictionary. Additionaly remember which base properties each wikidata_class had.
            if wikidata_class not in check_for_base_properties:
                check_for_base_properties[wikidata_class] = {}
                check_for_base_properties[wikidata_class]["class"] = [(seed_class, seed_label, seed_transitive_count, wikidata_class_label, transitive_wikidata_class_count)]
                check_for_base_properties[wikidata_class]["properties"] = []
            elif not any(row for row in check_for_base_properties[wikidata_class]["class"] if seed_class in row):
                check_for_base_properties[wikidata_class]["class"].append((seed_class, seed_label, seed_transitive_count, wikidata_class_label, transitive_wikidata_class_count))
            if prop in base_properties:    
                check_for_base_properties[wikidata_class]["properties"].append(prop)
            # skip if data_type no approved (quantity, time, or entity) or if property had been rejected, or if the wikidata_class had been rejected
            if (data_type not in white_list) or (prop in property_black_list) or (wikidata_class in class_black_list):
                continue
            # skip if no statistical information is available, and the datatype is not a quantity or time (datatypes that don't have statistical information)
            if not(tail_counts) and not(data_type in ["http://wikiba.se/ontology#Quantity","http://wikiba.se/ontology#Time"]):
                continue
            # if a property appears in less than 10% of the class occurences, skip unless property is one of the base properties
            if int(transitive_property_count) < 0.1 * int(float(transitive_wikidata_class_count)) and prop not in base_properties:
                continue
            if data_type == "http://wikiba.se/ontology#Quantity":
                constraint_label_list = ["QUANTITY"]
                constraint_class_list = ["QUANTITY"]
            if data_type == "http://wikiba.se/ontology#Time":
                constraint_label_list = ["TIMEX"]
                constraint_class_list = ["TIMEX"]


            if wikidata_class == seed_class:
                set_of_accepted_classes.update([wikidata_class])
                D.append({"seed class": seed_class, "seed label": seed_label, "transitive_cnt":seed_transitive_count, "type_label": wikidata_class_label, "type": wikidata_class, "transitive_class_instances_cnt": transitive_wikidata_class_count, "property": prop,\
            "property_label": property_label, "count": transitive_property_count, "also_known":known_as,"data_type":data_type,"constraint_label_list":constraint_label_list, "constraint_class_list":constraint_class_list,"tail_cnt":tail_counts})
            
            elif (int(float(transitive_wikidata_class_count)) >= 100) or (int(float(transitive_wikidata_class_count)) >= 0.05*int(float(seed_transitive_count))): 
                set_of_accepted_classes.update([wikidata_class])
                D.append({"seed class": seed_class, "seed label": seed_label, "transitive_cnt":seed_transitive_count, "type_label": wikidata_class_label, "type": wikidata_class, "transitive_class_instances_cnt": transitive_wikidata_class_count, "property": prop,\
            "property_label": property_label, "count": transitive_property_count, "also_known":known_as,"data_type":data_type,"constraint_label_list":constraint_label_list, "constraint_class_list":constraint_class_list,"tail_cnt":tail_counts})
    # for each of the accepted wikidata classes that were not filtered out, check whether they have missed out on any base properties and add them
    for wikidata_class in check_for_base_properties:
        if wikidata_class not in set_of_accepted_classes:
            continue
        #for prop in list(set(check_for_base_properties[wikidata_class]["properties"]) ^ set(base_properties)):
        missed = list(set(base_properties) - set(check_for_base_properties[wikidata_class]["properties"]))
        bases = set(check_for_base_properties[wikidata_class]["properties"])
        for prop in missed:
            if prop in time_properties:
                known_as = "Time"
                data_type = "http://wikiba.se/ontology#Time"
                constraint_label_list = ["TIMEX"]
                constraint_class_list = ["TIMEX"]
            elif prop in location_properties:
                known_as = "Location"
                data_type = "http://wikiba.se/ontology#WikibaseItem"
                constraint_class_list = ["Q56061", "Q515", "Q6256", "Q2221906","Q17334923"]
                constraint_label_list = ["administrative territorial entity","city","country","geographic location","location"]
            elif prop == "P1132":
                known_as = "number of participants"
                data_type = "http://wikiba.se/ontology#Quantity"
                constraint_label_list = ["QUANTITY"]
                constraint_class_list = ["QUANTITY"]
            else:
                known_as = "suspect, player, participants, event participant, competitor, between, belligerents, attendee, party, accused, agent"
                data_type = "http://wikiba.se/ontology#WikibaseItem"
                constraint_class_list = ['Q5', 'Q4539', 'Q58778', 'Q124964', 'Q170584', 'Q178651', 'Q321839', 'Q500834', 'Q1190554', 'Q3249551', 'Q13406554', 'Q14136353', 'Q16334295', 'Q24336466', 'Q54933017', 'Q70990126', 'Q5', 'Q4539', 'Q58778', 'Q124964', 'Q170584', 'Q178651', 'Q321839', 'Q500834', 'Q1190554', 'Q3249551', 'Q13406554', 'Q14136353', 'Q16334295', 'Q24336466', 'Q54933017', 'Q70990126']
                constraint_label_list = ['human', 'cooperative', 'system', 'coalition', 'project', 'interview', 'agreement', 'tournament', 'occurrence', 'process', 'sports competition', 'fictional event', 'group of humans', 'mythical event', 'race result', 'creative work season', 'fictional event', 'system', 'race result', 'occurrence', 'human', 'tournament', 'mythical event', 'interview', 'group of humans', 'agreement', 'coalition', 'project', 'sports competition', 'cooperative', 'creative work season', 'process']
            property_label = prop_labels[prop]

            for seed_row in check_for_base_properties[wikidata_class]["class"]:
                seed_class, seed_label, seed_transitive_count, wikidata_class_label, transitive_wikidata_class_count = seed_row
                D.append({"seed class": seed_class, "seed label": seed_label, "transitive_cnt":seed_transitive_count, "type_label": wikidata_class_label, "type": wikidata_class, "transitive_class_instances_cnt": transitive_wikidata_class_count, "property": prop,\
                "property_label": property_label, "count": "-1", "also_known":known_as,"data_type":data_type,"constraint_label_list":constraint_label_list, "constraint_class_list":constraint_class_list,"tail_cnt":"-1"})

    with open(path2, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["seed class", "seed label", "transitive_cnt", "type_label", "type", "transitive_class_instances_cnt", "property","property_label",\
                "count","also_known","data_type","constraint_label_list", "constraint_class_list","tail_cnt"])
        writer.writeheader()
        for row in D:
            writer.writerow(row)


def transfer_properties():
    '''
    Take properties from all wikidata classes and pass them to their superclasses
    '''
    all_subclasses = dict() # subclasses of each seed class
    proprow = dict() # row of the first occurence of a given property
    seeds = set()
    D = []
    with open(processing_sheets_path+"/reduced_data.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:   
            #row[0] = eval(row[0])[0]
            seed_class = row[0]  
            seeds.update([seed_class])
            wikidata_class = row[3]
            prop = row[5]
            row[7] = int(row[7])
            row[10] = eval(row[10]) if row[10] else row[10]
            row[11] = eval(row[11]) if row[11] else row[11]
            row[12] = eval(row[12]) if row[12] else row[12]
            row[12] = {i.split(":")[0].split("_")[-1]:int(i.split(":")[-1]) for i in row[12]}
            # change from "['square_Q174782:2', 'archaeological site_Q839954:1', 'street_Q79007:1']" to {'square_Q174782': 2, 'archaeological site_Q839954': 1, 'street_Q79007': 1}


            if seed_class not in all_subclasses:
                all_subclasses[seed_class] = set()
            all_subclasses[seed_class].update([wikidata_class])
            
            if seed_class not in proprow:
                proprow[seed_class] = dict()

            if prop not in proprow[seed_class]:
                proprow[seed_class][prop] = row
                proprow[seed_class][prop][2] = seeds_names[seed_class]
                proprow[seed_class][prop][3] = seed_class
                

            else:
                proprow[seed_class][prop][7] += int(row[7])
                c = Counter(proprow[seed_class][prop][12])
                c.update(row[12])  
                proprow[seed_class][prop][12] = dict(c)

            #if wikidata_class not in classprop:
                #classprop[wikidata_class]=[]
            #classprop[wikidata_class].append(prop)

            #if wikidata_class in proprow and prop in proprow[wikidata_class]:
                #proprow[wikidata_class].pop(prop, None)

    with open(processing_sheets_path+"/reduced_data.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data: 
            # we skip rows where the wikidata type is a seed class and write them in later
            row[10] = eval(row[10]) if row[10] else row[10]
            row[11] = eval(row[11]) if row[11] else row[11]
            row[12] = eval(row[12]) if row[12] else row[12]
            row[12] = {i.split(":")[0].split("_")[-1]:int(i.split(":")[-1]) for i in row[12]}
            if row[3] in seeds:
                continue
            else:
                D.append({"ACE": row[0],"transitive_cnt": row[1], "type_label": row[2], "type": row[3], "#type_instances": row[4], "property": row[5],\
                    "property_label": row[6], "count": row[7], "also_known":row[8],"data_type":row[9],"constraint_label_list":row[10]\
                        ,"constraint_class_list":row[11],"tail_cnt":row[12]})
        # We add rows for each property and transitive property of a seed class 
        for seed_class in proprow:
            for prop in proprow[seed_class]:
                row = proprow[seed_class][prop]
                D.append({"ACE": row[0],"transitive_cnt": row[1], "type_label": row[2], "type": row[3], "#type_instances": row[4], "property": row[5],\
            "property_label": row[6], "count": row[7], "also_known":row[8],"data_type":row[9],"constraint_label_list":row[10]\
                ,"constraint_class_list":row[11],"tail_cnt":row[12]})

    with open(processing_sheets_path+"/extended_prop_data.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["ACE", "transitive_cnt", "type_label", "type", "#type_instances", "property", "property_label", "count",\
         "also_known", "data_type", "constraint_label_list", "constraint_class_list", "tail_cnt"])
        writer.writeheader()
        for row in D:
            writer.writerow(row)
                

def generate_questions(path = processing_sheets_path+"/final_sheet.csv"):
    new = []
    D = {}
    with open(processing_sheets_path+"/property_questions.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:          
            D[row[0]] = row[2].split("\n")

    with open(processing_sheets_path+"/reduced_data.csv", "r") as csvfile:
        next(csvfile)
        data = csv.reader(csvfile, delimiter=",")
        for row in data:
            seed_class = row[0]  
            seed_label = row[1]
            seed_transitive_count = row[2]
            wikidata_class_label = row[3]
            wikidata_class = row[4]
            transitive_wikidata_class_count = row[5]
            property = row[6]
            property_label = row[7]
            transitive_property_count = row[8]
            known_as = row[9]
            data_type = row[10]
            constraint_label_list = row[11]
            constraint_class_list = row[12]
            tail_counts = row[13]      
            template_questions = D[property]
            questions = []

            if not template_questions or template_questions == [] or template_questions == "" or template_questions=="[]":
                continue
            for question in template_questions:
                if question:
                    question = question.replace("@@", wikidata_class_label)
                    questions.append(question)
            new.append({"seed class": seed_class, "seed label": seed_label, "transitive_cnt":seed_transitive_count, "type_label": wikidata_class_label, "type": wikidata_class, "transitive_class_instances_cnt": transitive_wikidata_class_count, "property": property,\
            "property_label": property_label, "count": transitive_property_count, "also_known":known_as,"data_type":data_type,"constraint_label_list":constraint_label_list, "constraint_class_list":constraint_class_list,"tail_cnt":tail_counts, "questions":questions})

    with open(path, "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["seed class", "seed label", "transitive_cnt", "type_label", "type", "transitive_class_instances_cnt", "property","property_label",\
                "count","also_known","data_type","constraint_label_list", "constraint_class_list","tail_cnt","questions"])
        writer.writeheader()
        for row in new:
            writer.writerow(row)

create_sheet() #data.csv
reduce_number_of_properties() #reduced_data.csv
#transfer_properties()           
generate_questions() #final_sheet.csv

