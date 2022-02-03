from utils.fileGen import fileGen



FEATURE_PATH = "/content/PathLossPredictionSatelliteImages/Data_Folder/Feature_Matrix.csv"
OUTPUT_PATH = "/content/PathLossPredictionSatelliteImages/Data_Folder/Output_Matrix.csv"
IMAGE_PATH = "/content/PathLossPredictionSatelliteImages/Data_Folder/Height_Images_2_resized"
tofile = True
if tofile:
    file_generator = fileGen(FEATURE_PATH, OUTPUT_PATH)
    file_generator.generate_files(root_dir='/content/PathLossPredictionSatelliteImages/Data_Folder')