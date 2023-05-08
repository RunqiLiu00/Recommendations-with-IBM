import numpy as np
import pandas as pd
import recommender_functions as rf
import sys # can use sys to take command line arguments

class Recommender:
    def __init__(self, df_file, df_content_file):
        self.df = pd.read_csv(df_file)
        self.df_content = pd.read_csv(df_content_file)
        self.df['user_id'] = rf.email_mapper(self.df)
        del self.df['email']
        self.df_content = self.df_content.drop_duplicates(subset=['article_id'], keep='first')
        self.df_content.rename(columns={'doc_full_name': 'title'}, inplace=True)
        self.user_item = rf.create_user_item_matrix(self.df)
        self.similarity_matrix = rf.get_article_similarity(self.df_content)

    
    # Make Knowledge-Based Recommendations
    def get_top_articles(self,n):
        '''
        INPUT:
        n - (int) the number of top articles to return
        
        OUTPUT:
        top_ids - (list) A list of the top 'n' article ids (string)
        top_articles - (list) A list of the top 'n' article titles 
        
        '''
        top_ids = list(self.df['article_id'].value_counts().iloc[:n].index)
        
        top_articles = []
        for id in top_ids:
            title = self.df.loc[self.df['article_id']==id]['title'].iloc[0]
            top_articles.append(title)  
        
        return top_ids, top_articles 
    
    # Make User-User Based Collaborative Filtering Recommendations
    def get_user_user_recs(self, user_id, m=10):
        '''
        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommendations you want for the user

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title
        '''

        if user_id not in self.user_item.index:
            return "Unknown User"
        else:
            articles_read_ids, _= rf.get_user_articles(user_id, self.user_item, self.df)
            neighbors_df = rf.get_top_sorted_users(user_id,self.df, self.user_item)

            recs = np.array([])

            for user in neighbors_df['neighbor_id']:
                read_ids, _ = rf.get_user_articles(user,self.user_item, self.df)

                new_recs = np.setdiff1d(read_ids, articles_read_ids, assume_unique=True)

                recs = np.concatenate([new_recs, recs], axis=0)
                _, idx = np.unique(recs, return_index=True)
                recs = recs[np.sort(idx)]
                if len(recs) > (m - 1):
                    break
            recs = recs[:m]
            rec_names = rf.get_article_names(recs.tolist(), self.df)

        return recs, rec_names
    
    def get_similar_articles(self, article, n=2):
        '''
        INPUT:
        article - (int or str) article ID or article name
        n - (int) the number of similar articles you want for the article
                
        OUTPUT:
        similar_articles_df - (pandas dataframe) a dataframe with:
                        article_id - is a neighbor article_id
                        similarity - measure of the similarity of each article_id to the provided article_id
        '''
        if isinstance(article, int) or isinstance(article, float):
            article_id = article
        elif isinstance(article, str):
            article_id = self.df_content.loc[self.df_content['title'] == article, 'article_id'].values[0]
        else:
            raise ValueError("Invalid article input. Please provide an article ID or article name.")
        
        if article_id not in self.df_content['article_id'].values:
            return None
        index = self.df_content[self.df_content['article_id'] == article_id].index.item()
        similarity_scores = self.similarity_matrix[index]

        similar_articles_df = pd.DataFrame({
            'article_id': self.df_content['article_id'],
            'title': self.df_content['title'],
            'similarity': similarity_scores
        })
        
        similar_articles_df = similar_articles_df.sort_values(by='similarity', ascending=False)
        similar_articles_df = similar_articles_df[similar_articles_df['article_id'] != article_id]
        similar_articles_df = similar_articles_df.reset_index(drop=True)
        
        return similar_articles_df[:n]
    
    # Make Content-Based Recommendations
    def content_recs(self,user_id, m=10, n=2):
        '''
        INPUT:
        user_id - (int) a user id
        m - (int) the number of recommended articles 
        n - (int) an input for get_article_names()  

        OUTPUT:
        recs - (list) a list of recommendations for the user by article id
        rec_names - (list) a list of recommendations for the user by article title
        '''
        user_articles,_ = rf.get_user_articles(user_id,self.user_item, self.df)
 
        recs = np.array([])
    
        for article in user_articles:
            similar_articles = self.get_similar_articles(article,n)

            if similar_articles is None:
                continue
            similar_articles = similar_articles['article_id']
            
            new_recs = np.setdiff1d(similar_articles, user_articles, assume_unique=True)

            recs = np.concatenate([new_recs,recs], axis=0)
            _, idx = np.unique(recs,return_index=True)
            recs = recs[np.sort(idx)]
            if len(recs)>(m-1):
                break
        recs = recs[:m] 
        rec_names = rf.get_article_names(recs.tolist(),self.df_content) if len(recs)>0 else []
        return recs, rec_names

# Demo
if __name__ == '__main__':

    #instantiate recommender
    rec = Recommender("user-item-interactions.csv","articles_community.csv")
    # make recommendations
    _, top_articles = rec.get_top_articles(10)
    _, user_user_recs = rec.get_user_user_recs(1, m=10)
    _, content_recs = rec.content_recs(1, m=10)

    print("The top {} popular articles are:".format(len(top_articles)))
    for article in top_articles:
        print('     '+article)
    print(" ")

    print("Based on user-user collaborative filtering, the recommened articles for user 1 are:") 
    for article in user_user_recs:
        print('     '+article)
    print(" ")

    print("Based on article descriptions, the recommened articles for user 1 are:") 
    for article in content_recs:
        print('     '+article)
    print(" ")
 
    print("Based on article descriptions, these are 2 articles the most similar to 'Data Wrangling at Slack'")
    print(rec.get_similar_articles('Data Wrangling at Slack', n=2))
