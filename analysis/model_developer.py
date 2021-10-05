import boto3
import pandas as pd
from gensim.models import Word2Vec, TfidfModel
import numpy as np

from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import re,os,glob
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial

import gensim


def make_w2v(df):
    speeches = [i.split() for i in df.speech_processed]
    w2v = Word2Vec(speeches,window=10,sg=1,workers=8)
    return w2v

def term_rankings(H,terms):
    term_rankings = []
    for topic_index in range(H.shape[0]):
        top_indices = np.argsort(H[topic_index,:])[::-1]
        term_ranking = [terms[i] for i in top_indices[:20]]
        term_rankings.append(term_ranking)
    return term_rankings

def similarity( w2v, ranking_i, ranking_j ):
    sim = 0.0
    pairs = 0
    for term_i in ranking_i:
        for term_j in ranking_j:
            try:
                sim += w2v.wv.similarity(term_i, term_j)
                pairs += 1
            except:
                pass
    if pairs == 0:
        return 0.0
    return sim/pairs

def tc_w2v(term_rankings,w2v):
    topic_scores = []
    overall = 0
    for index, topic in enumerate(term_rankings):
        score = similarity(w2v,topic,topic)
        topic_scores.append(score)
        overall += score
    overall /= len(term_rankings)
    return overall

def run_coherence(H,terms,w2v):
    rankings = term_rankings(H,terms)
    coherence = tc_w2v(rankings,w2v)
    return coherence

def run_NMF_model(k,data):
    model = NMF(n_components=k,
                             init='nndsvd',
                             max_iter=500,
                             random_state=1234)

    W = model.fit_transform(data['dtm'])
    H = model.components_
    TC_W2V = run_coherence(H,data['vocab'],data['w2v'])

    return {"W":W,'H':H,"model":model,'TC_W2V':TC_W2V,'k':k,'terms':data['vocab']}


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

class model_developer():
    def __init__(self):
        self.connection = boto3.client('s3')
        self.best_k = None
        
    def load_dataframe(self,S3Bucket,path,party=['D','R'],chambers=['H','S'],years = None,verbose=False):
        df = pd.read_csv(self.connection.get_object(Bucket=S3Bucket,
                                        Key = path)['Body'])
    
        if verbose:
            print(f"# speeches pre-duplicate removal -- {len(df)}")


        # filter house speeches
        if type(chambers) != list:
            chambers = [chambers]
        df = df.loc[(df.chamber_x.isin(chambers))]
        
        if type(party) != list:
            party = [party]
        df = df.loc[df.party.isin(party)]
        
        if years:
            if type(years) != list:
                years = [years]
            df = df.loc[df.year.isin(years)]
        

        if verbose:
            print(f"# Speeches after filtering {years}, {chambers}, {party} -- {len(df)}")

        self.df = df
    
    def make_w2v(self):
        self.w2v = make_w2v(self.df)
    
    def make_dtm(self,min_df=2,max_df=0.25):
        self.tfidf = TfidfVectorizer(min_df = min_df, max_df = max_df,stop_words=procedural_stop_words)
        self.dtm = self.tfidf.fit_transform(self.df['speech_processed'])
        self.vocab = self.tfidf.get_feature_names()

    def make_gensim(self,min_df=2,max_df=0.25):
        speeches = [[word for word in speech.split() if word not in procedural_stop_words] for speech in self.df.speech_processed]
        self.speeches = speeches
        self.dictionary = gensim.corpora.Dictionary(speeches)
        self.dictionary.filter_extremes(no_below=min_df,no_above=max_df)
        self.corpus = [self.dictionary.doc2bow(text) for text in speeches]
        model = TfidfModel(self.corpus)
        self.tfidf = [model[i] for i in self.corpus]
        
        
    
    def Run(self,k_range,plot=False):
        with Pool(len(k_range)) as p:
            run = partial(run_NMF_model,data={'dtm':self.dtm,'vocab':self.vocab,'w2v':self.w2v})
            models = p.map(run,k_range)
        
        self.model_storage = models
        val_df = pd.DataFrame([{'k':x['k'],"TC-W2V":x['TC_W2V']} for x in self.model_storage])
        
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            k_max = val_df.loc[val_df['TC-W2V'] == val_df['TC-W2V'].max()]

            ax.plot(val_df['k'],val_df['TC-W2V'])
            ax.set_xlabel('k')
            ax.set_ylabel('TC-W2V')

            print_val = f"{k_max['k'].values[0]}: {np.round(k_max['TC-W2V'].values[0],3)}"
            ax.annotate('%s'%print_val, xy=(k_max['k'], k_max['TC-W2V']), xytext=(k_max['k'], k_max['TC-W2V']-0.01),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        )
            plt.show()
    
    
    
    
if __name__ == "__main__":
    print('loaded model_developer.py')
#     tester = model_developer()
#     tester.load_dataframe('ascsagemaker',
#                           'JMP_congressional_nmf/House_bigrams/101.csv',
#                           chambers='S',
#                           years=[1989],
#                           verbose=True)
#     tester.make_w2v()
#     tester.make_dtm()