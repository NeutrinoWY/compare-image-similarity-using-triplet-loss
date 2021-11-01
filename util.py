import os
import shutil
from datetime import datetime
from zipfile import ZipFile


def createDirectory(dirPath, verbose = True):
    """
    if directory not exist, create one. 
    """
       
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)
        if verbose:
            print("Folder not found!!!   " + dirPath + " created.")


def createCheckpointDir(outputFolderPath = '../experiments/', debug_mode = False):
    
    ## Create output folder, if it does not exist
    createDirectory(outputFolderPath, verbose = False)
    
    ## Create folder to save current version, if it does not exist
    if debug_mode:
        outputCurrVerFolderPath = os.path.join( outputFolderPath, 'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_debug' )
    else:
        outputCurrVerFolderPath = os.path.join( outputFolderPath,  'Checkpoint_ver_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') )
        
        
    createDirectory(outputCurrVerFolderPath, verbose = False)
    print("Output will be saved to:  " + outputCurrVerFolderPath)
    
    return outputCurrVerFolderPath

        


