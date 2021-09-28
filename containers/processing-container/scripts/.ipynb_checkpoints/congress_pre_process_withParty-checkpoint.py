import os, re, joblib,json
import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex
import spacy
nlp = spacy.load("en_core_web_sm",exclude=['parser','ner','entity_linker',
                                            'entity_ruler','textcat',
                                            'textcat_multilabel','senter','sentencizer',
                                            'morphologozier','transformer'])

import requests
from bs4 import BeautifulSoup

from string import punctuation
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from optparse import OptionParser
from gensim.models.phrases import Phrases
import time


def omit_senate_special_language(x):
    language = ["to pay tribute", "honor the memory", "when the senate completes its business today",
                "stand in recess", "wave the point of order", "make a point of order",
                "raise a point of order"]

    captured = [token for token in language if x.find(token) >= 0]
    return False if captured else True

def remove_phrases(speech,phrases):
    for phrase in phrases:
        speech.replace(phrase,'')
    return speech

def merge_texts(corpus_list):
    Texts = []
    for corpus in corpus_list:
        Texts.extend(pd.read_csv(corpus)['speech_processed'])
    Texts = [text.split() for text in Texts]
    return Texts


def custom_tokenizer(nlp):
    """
    custom spacy tokenizer for maintaining hyphenated words
    """
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = compile_suffix_regex(nlp.Defaults.suffixes)

    return Tokenizer(nlp.vocab, prefix_search=prefix_re.search,
                                suffix_search=suffix_re.search,
                                infix_finditer=infix_re.finditer,
                                token_match=None)

nlp.tokenizer = custom_tokenizer(nlp)


def POS_select(speech):
    """
    speech tokenizer, lemmatizes only NOUN, PROPN, VERB, ADJ
    """
    # Tokenize and lemmatize
    text = []
    for token in speech:
        if token.pos_ in ['NOUN','PROPN','VERB','ADJ']:
            text.append(token.lemma_.lower().replace('.',''))
    return text

def chamber_table(states,chamber,cong):
    """
    Gets the party ID of every member of congress for a given congress
    """
    rows = []
    if cong < 111:
        first_tag = 'ul'
        second_tag = 'li'
    else:
        first_tag = 'dl'
        second_tag = 'dd'

    rows = []
    for td in states:
        for element in td.find_all(['h4',first_tag]):
            if element.name == 'h4':
                state = element.text.replace('[edit]','')
            else:
                for name in element.find_all(second_tag):
                    if 'vacant' not in name.text.lower():
                        sen_name = name.find_all('a')[-1].text
                        sen_name = sen_name.replace('.Jr','').strip().upper()
                        last_name = sen_name.split()[-1]
                        party = re.findall('\(([aA-zZ])',name.text)[0]
                        rows.append({'state':state,'name':sen_name,'party':party,'chamber':chamber,'last_name':last_name})
    df = pd.DataFrame(rows)
    df = df.groupby(['state','name']).first().reset_index()
    return(df)

class speech_processor():
    """
    class for generating and processing speeches
    """
    def __init__(self,path,chamber):
        self.chamber = chamber
        self.path = path
        self.speeches = None
        self.df = None

    def generate_speeches_df(self,wc=50,start_date=None,end_date=None,testing=False):
        """
        parse speeches, link to meta-data, and filter

        args:
            - wc: minimum speech length
            - start_date: first date in range
            - end_date: last date in range
        """

        # load in speeches
        speeches = open(os.path.join(self.path,'speeches',f"speeches_{self.chamber}.txt"),
                        encoding='utf-8',
                        errors='ignore').read().split('\n')
        #speeches = {row[:10]:row[11:].strip() for row in speeches}
        #speeches = dict([row.strip().split('|') for row in speeches])

        speeches = { row[:row.index('|')]:row.strip()[row.index('|')+1:] for row in speeches if row.count('|')>0 }
        
        
        print(f'{len(speeches)} speeches in speeches_{self.chamber}.txt')
        
        # get description file processed
        congress = open(os.path.join(self.path,'descr',f"descr_{self.chamber}.txt"),
                         encoding='utf-8',
                         errors='ignore').read()
        rows = [r.split("|") for r in congress.split('\n')[1:-1]]
                 
        print(f'{len(rows)} rows in descr_{self.chamber}.txt')

                 
        columns = congress.split('\n')[0].split('|')
        df = pd.DataFrame(rows, columns=columns)

        # link data
        df['speech_text'] = df.speech_id.apply(lambda x: speeches[x])
        df = df.groupby('speech_text').first().reset_index() # duplicates exist
        df['date'] = pd.to_datetime(df.date)

        # subset by date range
        if start_date:
            df = df.loc[df.date >= start_date]
        if end_date:
            df = df.loc[df.date < end_date]

        # filter data
        self.df = df.loc[(df.gender != "Special") &
                        (df.chamber != 'E') &
                        (df.gender != 'Unknown') &
                        (df.word_count.astype(int) >= wc) &
                        (df.speech_text.apply(omit_senate_special_language))]

        if testing:
            print('testing: only using 500 speeches')
            self.df = self.df.sample(500)

    def process_speeches(self, omit_path = None, min_df = 50, threshold = 10,
                        batch_size=20):
        """
        pre-process speeches. Performs normalization, phrase removal, tokenization,
        and bigram collocation.

        args:
            - omit_path: path to .csv file containing tokens to remove
            - min_df: minimum number of documents for collocation
            - threshold: collocation threshold

        """

        if omit_path:
            omit_tokens = open(omit_path,'r').read().split(',')
        else:
            omit_tokens = []

        # normalize text
        text_normalized = self.df['speech_text'].str.lower().str.translate(punctuation)

        # remove phrases
        text_phrased = [remove_phrases(speech,omit_tokens) for speech in tqdm(text_normalized,desc="omitting phrases")]

        # tokenize
        text_tokenized = nlp.pipe(text_phrased,batch_size=batch_size, n_process=4)
        print('text spacyfied')
        text_tokens = [POS_select(speech) for speech in tqdm(text_tokenized)]

        bigrams = Phrases(text_tokens, min_count=min_df, threshold=threshold)
        speech_ngrams = [bigrams[sent] for sent in text_tokens]
        
        # rejoin
        self.df['speech_processed'] = [' '.join(speech) for speech in speech_ngrams]


    
    
    def make_speech_list(self):
        
        H = self.df.loc[(self.df.chamber == 'H')]
        S = self.df.loc[(self.df.chamber == 'S')]
        html_text = f'https://en.wikipedia.org/wiki/{self.chamber}th_United_States_Congress#Members'
        print(html_text)
        html = BeautifulSoup(requests.get(html_text).text,'html.parser')
        h_states = html.find('span',{'id':'House_of_Representatives_3'}).find_next('table',class_='multicol').find('tr').find_all('td')
        s_states = html.find('span',{'id':'Senate_3'}).find_next('table').find('tr').find_all('td')[:2]

        s_df = chamber_table(s_states,'S',int(self.chamber))
        h_df = chamber_table(h_states,'H',int(self.chamber))
        party_df = pd.concat([s_df,h_df])

        df = self.df.merge(party_df,on=['last_name'],how='inner')
        df['year'] = pd.to_datetime(df.date).dt.year
        df = df.loc[df.party.isin(['D','R'])]

        self.df = df.groupby('speech_id').first().reset_index()
        


    def save_speeches_df(self,out_path,filename=None):
        """
        save speeches pandas DataFrame

        args:
            - out_path: directory to save data
            - filname: optional filename

        """
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        if not filename:
            filename = f"{self.chamber}.csv"
        else:
            if not filename.endswith('.csv'):
                raise Exception("File must be csv")

        self.df.to_csv(os.path.join(out_path,filename))

def main(path, chamber, out_path, wc=50, start_date=None, end_date = None,
         ngram_min_df = 50, ngram_thresh=10, batch_size=20,omit_path = None,
         dtm_min_df = 2, dtm_max_df = 0.25, filename=None,testing=False):

    processor = speech_processor(path,chamber)
    processor.generate_speeches_df(wc,start_date,end_date,testing)
    print('parsed speeches')
    processor.process_speeches(omit_path,ngram_min_df,ngram_thresh,batch_size)
    print('pre-processed speeches')
    processor.make_speech_list()
    processor.save_speeches_df(out_path,filename)

if __name__ == '__main__':
    parser = OptionParser(usage="usage: %prog [options] chamber")
    parser.add_option('--wc',action='store',type='int',dest='wc',default=50)
    parser.add_option('--sd',action='store',type='string',dest='start_date',default=None)
    parser.add_option('--ed',action='store',type='string',dest='end_date',default=None)
    parser.add_option('--o',action='store',type='string',dest='omit_path',default='/opt/ml/input/omit_phrases.csv')
    parser.add_option('--nmin',action='store',type='int',dest='ngram_min_df',default=50)
    parser.add_option('--nt',action='store',type='int',dest='ngram_thresh',default=10)
    parser.add_option('--b',action='store',type='int',dest='batch_size',default=20)
    parser.add_option('--mindf',action='store',type='int',dest='dtm_min_df',default=2)
    parser.add_option('--maxdf',action='store',type='float',dest='dtm_max_df',default=.25)
    parser.add_option('--f',action='store',type='string',dest='filename',default=None)
    parser.add_option('--t',action='store_true', dest='testing')
    (options,args) = parser.parse_args()
    path = '/opt/ml/processing/input'
    out_path = '/opt/ml/processing/output'
    chamber = args[0]

    main(path,chamber,out_path,options.wc, options.start_date,
        options.end_date,options.ngram_min_df,options.ngram_thresh,options.batch_size,
        options.omit_path,options.dtm_min_df,options.dtm_max_df,options.filename,
        options.testing)
