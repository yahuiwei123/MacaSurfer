import os
print("Setting environment ...")
os.environ['nnUNet_raw_data_base'] = "../nnUNet_raw_data_base"
os.environ['nnUNet_raw_data'] = "../nnUNet_raw_data_base/nnUNet_raw_data"
os.environ['nnUNet_raw'] = "../"
os.environ['RESULTS_FOLDER'] = "../nnUNet_trained_models"
os.environ['nnUNet_preprocessed'] ="../nnUNet_preprocessed"
os.environ['PARAM_SEARCH_FOLDER'] ="../nnUNet_param_search_folder"