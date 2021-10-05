import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import tarfile, io, joblib
import boto3

client = boto3.client('s3')
paginator = client.get_paginator('list_objects_v2')


def reliability(model1,model2):
    # get the number of topics
    k = model1.get_topics().shape[0]

    # get distributions
    W1 = model1.get_topics()
    W2 = model2.get_topics()

    tv = []
    for i1 in range(k):
        matches = []
        m1_terms = model1.get_topic_terms(i1)
        for i2 in range(k):
            m2_terms = model2.get_topic_terms(i2)
            vals1 = []
            vals2 = []
            for ix in set([_[0] for _ in m1_terms] + [_[0] for _ in m2_terms]):
                vals1.append(W1[i1,ix])
                vals2.append(W2[i2,ix])

            if 1 - cosine(vals1,vals2) >= 0.7:
                matches.append(i2)

        tv.append(matches)
    return sum([1 if len(x) else 0 for x in tv])/k



def calculate_reliability(model_obj):
    unique_k = np.unique([model['k'] for model in model_obj])
    K_vals = []
    
    for k in unique_k:
        subset_models = [model for model in model_obj if model['k'] == k]
        total_reliability = []
        for i1 in range(len(subset_models)):
            for i2 in range(len(subset_models)):
                if i1 != i2:
                    rel = reliability(subset_models[i1]['model'],subset_models[i2]['model'])
                    Row = {"k":k,'rel':rel}
                    K_vals.append(Row)
                    
    return K_vals


def extract_data(congress,prefix):
    # format model congress name
    cstr = '{:0>3}'.format(congress)
    
    # find the model output object
    for page in paginator.paginate(Bucket='ascsagemaker',Prefix=f"{prefix}/{congress}/"):
        for ob in page['Contents']:
            if ob['Key'].endswith('.tar.gz'):
                obj_name = ob['Key']
    
                # load the model from S3 
                object_ = client.get_object(Bucket='ascsagemaker',Key=obj_name)['Body'].read()
                tar = tarfile.open(fileobj=io.BytesIO(object_))

                model = joblib.load(tar.extractfile(member=tar.getmember(f"NMF_{cstr}_evaluations.pkl")))
                return model