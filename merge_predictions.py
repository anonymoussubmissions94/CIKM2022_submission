import os
import json 

def merge_predictions(directory="data/predictions/"):
    paths = []
    file_core = ''
    for subdirectory in sorted(os.listdir(directory)):
        for file in sorted(os.listdir(directory+subdirectory)):
            if not file_core:  
                predictions = {}
            if file_core and (not "_" in file or file_core not in file):
                with open("data/predictions/"+str(subdirectory)+"/"+file_core+".txt", "w") as f:
                    json.dump(predictions, f)
                file_core = ''
                for file_path in paths:
                    os.remove(directory+subdirectory+"/"+file_path)
                paths = []

            if "_" not in file:
                continue
            paths.append(file)
            with open(directory+subdirectory+"/"+file,"r") as f:
                quarter_predictions = json.load(f)
                file_core = file.split("_")[0]
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
    with open("data/predictions/"+str(subdirectory)+"/"+file_core+".txt", "w") as f:
        json.dump(predictions, f)
        for file_path in paths:
            os.remove(directory+subdirectory+"/"+file_path)
        paths = []
        

if __name__ == "__main__":
    merge_predictions()