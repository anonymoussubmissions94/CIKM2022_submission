from format_new_dataset import *
import os     

unprocessed_data_path = project_path + ""
for folder_input_path in os.listdir(unprocessed_data_path):
    output_path_sufix = folder_input_path.split("/")[-1]
    output_folder_path = project_path + "/processed_data/" + output_path_sufix
    os.mkdir(output_folder_path)
    output_file_path  = project_path + "/processed_data/" + output_path_sufix +"/" +  output_path_sufix +".txt"
    format_new_dataset(folder_input_path, output_file_path ,"ace-event")


