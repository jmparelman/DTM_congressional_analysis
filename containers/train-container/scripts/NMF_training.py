import joblib, os, json, argparse
from sklearn import decomposition

from multiprocessing import Pool
import time

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, nmf, CoherenceModel
from scipy.spatial.distance import cosine

import pandas as pd 
import numpy as np

base_path = '/opt/ml/'
input_path = os.path.join(base_path,'input/data')
output_path = os.path.join(base_path,'output')
model_path = os.path.join(base_path,f'model')

training_path = os.path.join(input_path,'training')

# read in the data, should be a single .csv file
training_file = os.listdir(training_path)[0]
df = pd.read_csv(os.path.join(training_path,training_file))

congress = training_file.split('.')[0]


procedural_stop_words = ['talk','thing','colleague','hear','floor','think','thank','insert','section','act_chair','amendment','clerk','clerk_designate',
                        'pursuant','minute','desk','amendment_text','amendment_desk','rule','debate','process','offer_amendment','majority','order',
                        'pass','extension','urge','urge_colleague','defeat_previous','yield_balance','member','committee','chairman','mr','subcommittee',
                        'rank_member','mr_chairman','oversight','yield_minute','yield_time','gentlewoman','gentleman','gentlelady','h_r','time_consume',
                        'legislation','measure','rollcall','rollcall_vote','vote_aye','vote_nay','nay','debate','point_order','chair','clause',
                        'clause_rule','germane','sustain','remark','conference','pass','oppose','offer','opposition','ask','speaker','bill',
                        'follow_prayer','approve_date','pledge_journal','morning_hour','today_adjourn','proceeding','deem_expire','reserve','complete',
                        'permit_speak','authorize_meet','session_senate','office_building','entitle','conduct_hearing','m_room','consent','ask_unanimous',
                        'dirksen_senate','senate_proceed','intervene_action','consider','notify_senate','senate','legislative_session','legislation',
                        'legislature','further_motion','motion','lay_table','motion_reconsider','reconsider','hearing','leader','p_m','a_m','period_morning',
                        'period_afternoon','executive_session','follow','senate_proceed','morning_business','authorize','motion_concur','concur','session',
                        'hour','control','follow_morning','senate_resume','follow','monday','tuesday','wednesday','thursday','friday','ask_unanimous',
                        'motion_reconsider','amendment','consent','motion_proceed','cloture','proceed','motion_invoke','cloture_motion','leader','invoke',
                        'no_','modify']


# filter house speeches
df = df.loc[df.chamber_x == 'H']
df = df.loc[df.party.isin(['R'])]

# make gensim dict and corpus, tfidf transform
speeches = [[word for word in speech.split() if word not in procedural_stop_words] for speech in df.speech_processed]

id2word = Dictionary(speeches)
id2word.filter_extremes(no_below=0.001*len(df),no_above=0.35)
corpus = [id2word.doc2bow(text) for text in speeches]
model = TfidfModel(corpus)
tfidf_corpus = [model[text] for text in corpus]


def Run_model(vals):
    
    model = nmf.Nmf(corpus = tfidf_corpus,
                    id2word = id2word,
                    num_topics = vals[0],
                    random_state = vals[1],
                    normalize = True,
                    passes = 20)
    
    coh_model = CoherenceModel(model = model,
                               texts=speeches,
                               dictionary = id2word,
                               coherence='c_v',
                               processes=1)
    
    return {"model":model,
            'seed':vals[1],
            'k':vals[0],
            'c_v':coh_model.get_coherence()}

Ks = range(10,110,10)
num_iters = 5
iter_list = []
for k in Ks:
    iter_list.extend([(k,np.random.randint(1,1000)) for _ in range(num_iters)])

with Pool(50) as p:
    output = p.map(Run_model,iter_list)
    
with open(os.path.join(model_path,f'NMF_{congress}_evaluations.pkl'),'wb') as out:
    joblib.dump(output,out)
