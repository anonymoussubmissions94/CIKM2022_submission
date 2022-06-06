# Hierarchical Sub-event Extraction

This repository present contains the code for the 2022 CIKM submission Extracting Semantic Sub-event Representations through Hierarchical Zero-shot Question Answering.

### Steps


To replicate paper results, run steps as follows.

* Run Follow instructions in the ```eventtextwikipediadumper``` to run [MWDumper](https://www.mediawiki.org/wiki/Manual:MWDumper) and get Wikipedia articles of events in .ndjson file format. Place the resulting files into ```data\event_articles```in the project folder.
* Run ```subevents/data_download.sh``` to prepare Wikidata and Dbpedia dumps and redirects.
* Run ```subevents/create_shelves.py``` to assure quick access to the dumps.  
* Set your project path in the ```config.ini```. 
* Run ```prepare_data.py``` to prepare and process data for event type prediction. This will create the data and the full folder structure as expected from subsequent code.
* Download the pretrained event detection model [here](https://ai2-s2-research.s3-us-west-2.amazonaws.com/dygiepp/master/ace05-event.tar.gz) and place it into ```pretrained/```.


#### Evaluation with Unlinked Sub-events
* Run ```scripts/unlinked_event_type_detection.sh``` to get predicted ACE-ontology event types on the data. This can take quite a while, the intermediate results will be stored in the data/intermediate_results folder. 
* Run ```merge_predictions.py``` to get the final results of event type detection.
* Run ```main.py``` to extract events from Wikipedia articles of events given predicted ACE-ontology event types. The results will be stored in the ```evaluation/unlinked_sub-events/results/table4.csv``` file.



#### Evaluation with Linked Sub-events
* Run ```scripts/linked_event_type_detection.sh```. 
* Run ```prepare_groundtruth.py``` to prepare prediction and evaluation on linked sub-events.
* Run ```linked_event_evaluation.py``` to run the baselines and our approach on the groundtruth data. The final evaluation results as described in the paper will be created in the ```evaluation/linked_sub-events/results/results.txt``` file.


#### Extra
* Check ```notebook/wikidata_extraction.ipynb``` to see how we get Wikidata event classes, properties, constraints and statistics. 
* Check ```processing_sheets/process_sheets.py``` to see how we filter Wikidata event classes, get transitive statistics and propagate properties through Wikidata event classes.
