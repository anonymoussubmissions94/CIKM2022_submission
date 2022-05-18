new_class_properties[wd_class]= {}

for row in rows:
    if wikidata_class not in class_properties:
        class_properties[wikidata_class] = {}
    if prop not in class_properties[wikidata_class]:
        class_properties[wikidata_class][prop] = [prop_label, prop_count, also_known, data_type, constraint_label_list, constraint_class_list, tail_cnt]

new_class_properties = {}
# go through all classes
for wd_class in wd_classes:
    proprow = {}
    # go through all subclasses of a class
    for subclass in subclasses[wd_class]:
        # go through all properties of a subclass and sum up the counts, the constraints and the tails
        for prop in subclasses:
            if prop not in proprow:
                proprow[prop] = class_properties[subclass][prop]
            else:
                proprow[prop][1] += class_properties[subclass][prop][1]
                proprow[prop][4] += class_properties[subclass][prop][4]
                proprow[prop][5] += class_properties[subclass][prop][5]
                proprow[prop][6] += class_properties[subclass][prop][6]
    for prop in class_properties[wd_class]:
        if prop in proprow:
            new_class_properties[wd_class] = class_properties[wd_class][prop][1] + proprow[prop][1]
        else:
            new_class_properties[wd_class] = class_properties[wd_class][prop][1]
        
for wd_class in new_class_properties:
    for seed in seeds[wd_class]:
        for prop in new_class_properties[prop]:
            D.append


