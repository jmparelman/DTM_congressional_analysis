import joblib, os, json, argparse
from sklearn import decomposition
from joblib import Parallel, delayed
from multiprocessing import Pool
import time

base_path = '/opt/ml/'
input_path = os.path.join(base_path,'input/data')
output_path = os.path.join(base_path,'output')
model_path = os.path.join(base_path,f'model')



training_path = os.path.join(input_path,'training')

# read in the data, should be a single .pkl file
training_file = os.listdir(training_path)[0]
input_dict = joblib.load(os.path.join(training_path,training_file))
X = input_dict['dtm']
terms = input_dict['vocab']
speech_id = input_dict['speech_id']

# get the chamber name
chamber = training_file.split('.')[0]

def run_model(k):
    print(f'{k}.k starting')
    start_time = time.time()
    model = decomposition.NMF(n_components=k,init='nndsvd',max_iter=1000,random_state=1234)
    W = model.fit_transform(X)
    H = model.components_
    model_dict = {"W":W,'H':H,"model":model,'terms':terms,'speech_id':speech_id}
    print(f'{k}.k finished -- {round((time.time() - start_time)/60),2}m')
    
    with open(os.path.join(model_path,f'NMF_{chamber}_{k}.pkl'),'wb') as out:
        joblib.dump(model_dict,out)  

if __name__ == "__main__":
    # Uses m1.m4.16xlarge instance for multiprocessing (k = 40-100)
    with Pool(64) as p:
        p.map(run_model,range(5,75,5))
