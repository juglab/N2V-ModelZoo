from n2v.models import N2V, N2VConfig
import numpy as np
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import urllib
import os
import zipfile

import argparse as ap

def n2v_2D_read_npz(source):
    # Download Data
    # create a folder for our data.
    if not os.path.isdir('./data'):
        os.mkdir('./data')

    # check if data has been downloaded already
    zipPath="data/SEM.zip"
    if not os.path.exists(zipPath):
        #download and unzip data
        data = urllib.request.urlretrieve(source, zipPath)
        with zipfile.ZipFile(zipPath, 'r') as zip_ref:
            zip_ref.extractall("data")
        
    
    data = np.load('./data/SEM.zip')
    return data['X'], data['X_val']
    

def train(source, model_name, basedir): 
    X, X_val = n2v_2D_read_npz(source)
    
    model = N2V(None, name=model_name, basedir=basedir)
    model.train(X, X_val)


if __name__ == "__main__":
    parser = ap.ArgumentParser(description='N2V Training.')
    parser.add_argument('--source', help='URl to the training data.')
    parser.add_argument('--model_name', help='N2V model name.')
    parser.add_argument('--basedir', help='Basedir of the model.')
    
    args = parser.parse_args()
    
    train(source=args.source, model_name=args.model_name, basedir=args.basedir)
    
