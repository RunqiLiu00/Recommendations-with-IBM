import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Map the user email to a user_id column and remove the email column
def email_mapper(df):
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

# User-User Based Collaborative Filtering
def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user item matrix 
    
    Description:
    Return a matrix with user ids as rows and article ids on the columns with 1 values where a user interacted with 
    an article and a 0 otherwise
    '''
    user_item = df.groupby(['user_id','article_id'])['article_id'].max().unstack()
    user_item = user_item.applymap(lambda x: 0 if pd.isna(x) else 1)
    return user_item 

def get_article_names(article_ids, df):
    '''
    INPUT:
    article_ids - (list) a list of article ids (int)
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    '''
    if not article_ids:
        return []
    
    article_names = []
    for article_id in article_ids:
        article_name = df.loc[df['article_id']==article_id, 'title'].iloc[0]
        article_name = article_name.replace('\nName: title, dtype: object','')
        article_names.append(article_name)
    return article_names 

def get_user_articles(user_id, user_item, df):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    df - (pandas dataframe) user-item interaction dataframe
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids (int)
    
    Description:
    Provides a list of the article_ids and article titles that have been seen by a user
    '''
    article_ids = list(user_item.columns[user_item.loc[user_id]>0])
    article_names = get_article_names(article_ids,df)
    
    return article_ids, article_names 

def get_top_sorted_users(user_id, df, user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
            1's when a user has interacted with an article, 0 otherwise
    
            
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user 
                    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
     
    '''
    neighbors_df=pd.DataFrame(columns=['neighbor_id','similarity','num_interactions'])
    neighbors_df['neighbor_id'] = user_item.index
    neighbors_df.index = neighbors_df['neighbor_id']
    neighbors_df['num_interactions'] = df.groupby('user_id')[['article_id']].count() 
    neighbors_df['similarity'] = user_item.dot(user_item.loc[user_id].T)
    neighbors_df = neighbors_df.sort_values(by=['similarity','num_interactions'], ascending = [False, False])
    neighbors_df = neighbors_df.drop(user_id) 
    return neighbors_df 

# Content-Based Recommendations
def get_article_similarity(df_content):
    '''
    INPUT:
    df_content - (pandas dataframe) the dataframe defined above
            
    OUTPUT:
    similarity_matrix - a matrix of similarity between each pair of article_id
    '''
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    preprocessed_desc = []
    for desc in df_content['doc_description']:
        desc=str(desc)

        tokens = word_tokenize(desc)

        filtered_tokens = [
            lemmatizer.lemmatize(token.lower()) for token in tokens if token.lower() not in stop_words
        ]

        clean_desc = ' '.join(filtered_tokens)
        preprocessed_desc.append(clean_desc)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_desc)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix



