from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import time
import os
import nltk
from keybert import KeyBERT
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
import json

app = Flask(__name__)

global statement2,statement, key_text1, key_text2, intents, searchTerm

intents= ''
searchTerm = ''

from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer("all-mpnet-base-v2")

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

kw_model = KeyBERT(model=sentence_model)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")


os.chdir("C:/Users//RAVI KUMAR//demo//deepspeechnew")

df = pd.read_csv('sentences1.csv')
statement2 = df.values.tolist()

df1 = pd.read_csv('product.csv')


def there_exists(terms):
    for term in terms:
        if term in statement:
            return True

def keyword_ext(keywords):
    global statement
    stop_words = 'english'
    #keywords = kw_model.extract_keywords(statement, keyphrase_ngram_range=(4,2), stop_words = None, use_maxsum=True,nr_candidates=10, top_n=3)
    
    #keywords = str(keyword_st)
    #print(f"keywords {keywords} are extracted")
    return keywords

#mean ppolin - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model output contain all embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def cosimilarity(key_text, text_corp):
    global statement, key_text1, key_text2
    key_text=str(key_text)
    key_text1 = key_text

    text_corp=list(text_corp)
    
    text_corp_tuple = []
    text_corp_tuple = [a_tuple[1] for a_tuple in text_corp]

    embeddings1 = sentence_model.encode(key_text, convert_to_tensor=True)


# Load model from Hugging face hub

    encoded_input = tokenizer(text_corp_tuple, padding=True, truncation=True, return_tensors='pt')

#  Compute token embeddings

    with torch.no_grad():
         model_output = model(**encoded_input)

#  Perform pooling. In this case, mean pooling
   
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #print(sentence_embeddings)
    
    top_k =1
    cos_scores=util.cos_sim(embeddings1, sentence_embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    print("Sentence:", key_text, "\n")
    print("Top", top_k, "most similar sentences in corpus:")
    for idx in top_results[0:top_k]:
        print(text_corp[idx][2], "(Score: %.4f)" % (cos_scores[idx]))
    key_text1=text_corp[idx][2]
    print(f'keywords {key_text1} cosine similarity')

    
    #key_text2 = kw_model.extract_keywords(key_text1, keyphrase_ngram_range=(3,4), stop_words = 'english', use_maxsum=True,nr_candidates=10, top_n=2)
    return key_text1

def comp_products(comp_search_term, comp_statement):

    comp_search_term=str(comp_search_term)
    comp_statement=str(comp_statement)
    #print(comp_search_term)
    #print(comp_statement)
    new_list = [comp_search_term, comp_statement]
    df2 = pd.DataFrame(new_list)
    tdf = df2.T
    # adding column name to the respective columns
    tdf.columns =['product', 'intent']
    df1 = pd.read_csv('product.csv')
    #merged_df = df1.merge(df2, how = 'inner', on = ['product', 'intent'])
    mergedStuff = pd.merge(df1, tdf, on=['product', 'intent'], how='inner')

    try:
             merged_df1 = mergedStuff.iloc[0,1]
             print(f"user said:{merged_df1}/n")

    except Exception as e:
             print("No matching records")
             return None
    return merged_df1

@app.route('/intent', methods=['POST'])
def intent_ref():
    WAKE = "hello"
    print("START")
    while True:
        statement = WAKE
#       statement = keyword_ext().lower()

        if statement.count(WAKE) > 0:
            text1 = request.form['text1']
            statement = keyword_ext(text1).lower()
            #statement = keyword_ext(text1).lower()

            keywords1 = remove_stopwords(statement)
            pos_tagged_sent = nltk.pos_tag(nltk.tokenize.word_tokenize(keywords1)) 
            nouns = [tag[0] for tag in pos_tagged_sent if tag[1] in('NNP','NN', 'NNS', 'JJ')]
            search_term = nouns 

            #print(f"search_term before {search_term} are extracted")
            if not search_term:
               search_term = None
            else:
               search_term = search_term

            statement2 = df.values.tolist()
            statement=cosimilarity(statement, statement2)
            print(f"statement {statement} are extracted")
            #Create the list of dictionary
            result = [{'intents': statement, 'Search_Term': search_term}]
            
            return jsonify(result)
            #return jsonify({'intents': statement, 'Search_Term': search_term})

if __name__=='__main__':
    app.run(host='0.0.0.0', port='5000')
