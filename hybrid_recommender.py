from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

class hybrid_recommender:
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
	# Author: Devendra Kumar Sahu
	# Email: devsahu99@gmail.com
    """
    This package usage multiple algorithms and parameters to accomodate different set of use cases.

    Parameters:
    ------------------------------------------------------------
    item_clusters: int
        The number of clusters for item matrix generation. This parameter can be tuned
    top_results: int
        Number of recommendations needed. Default value is 10
    ratings_weightage: int
        Weightage for user ratings score. Default is 1
    content_weightage: int
        Weightage for content score. Default is 1
    null_rating_replace: str
        Value to be used as replacement for missing ratings. Default is 'mean', other acceptable values are 'zero','one', and 'min'
    ------------------------------------------------------------
    Returns:
    
    DataFrame having top recommended results for the list of users
    ------------------------------------------------------------
    Approach:

    1. Create an instance of the hybrid recommender class
         mr = hybrid_recommender()

    2. Call fit method on the defined object by passing on ratings and content data
         mr.fit(ratings_df,content_df)

    3. Call the predict method
        recommended_df = mr.predict()
        
    ------------------------------------------------------------
    # Example

    ### Create Ratings DataFrame
        item_id = [1,7,9,10,12,2,4,6,8,10,12,3,6,9,12,14,10,13,12,14,11,2,5,7,8,9,10,12]
        user_id = [1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,5]
        rating = [4,5,2,3,5,2,3,2,3,4,4,5,1,2,3,1,2,4,5,3,5,3,1,3,5,3,5,3]
        ratings = pd.DataFrame({'user_id':user_id,'item_id':item_id,'rating':rating})

    ### Create Content DataFrame
        items = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        cols = ['col1','col2','col3','col4','col5']
        feats =[[1,0,0,1,1],
               [1,1,0,0,1],
               [0,1,1,0,0],
               [0,1,1,1,0],
               [1,0,1,1,1],
               [1,1,1,0,0],
               [0,1,0,1,0],
               [0,0,0,1,0],
               [0,1,1,0,0],
               [1,1,1,0,1],
               [0,0,0,1,1],
               [0,1,0,1,0],
               [0,1,1,0,1],
               [0,0,1,1,1],]
        item_df = pd.DataFrame(feats,index=items,columns=cols)

    **Ratings DataFrame**
        ratings.head()

    **Content DataFrame**
        item_df.head()

    ## Fitting and prediction
        **Creating the recommender object**
            my_recommender = hybrid_recommenders(item_clusters=4,top_results=5)

        **Fitting the data**
            my_recommender.fit(ratings,item_df)

        **Recommend for few users**
            my_recommender.predict([1,2,3])

        **Recommendations for All users**
            my_recommender.predict()

    """
    def __init__(self,item_clusters=50,top_results=10,ratings_weightage=1,content_weightage=1,null_rating_replace='mean'):
        self.__clusters = item_clusters
        self.__top = top_results
        self.__ratings_weightage = ratings_weightage
        self.__content_weightage = content_weightage
        self.__null_replace = null_rating_replace
        
    def __getItemPreds(self):
        items = self.__item_df.index
        self.__item_preds = pd.DataFrame(cosine_similarity(self.__item_df, self.__item_df),index=items,columns=list(items))

    def __getRatingsPreds(self, rating_df):
        replace_ind = self.__null_replace
        users = rating_df.index
        items = rating_df.columns
        if replace_ind == 'zero':
            rating_df = rating_df.fillna(0)
        elif replace_ind == 'one':
            rating_df = rating_df.fillna(1)
        elif replace_ind == 'min':
            rating_df = rating_df.fillna(rating_df.min())
        else:
            rating_df = rating_df.fillna(rating_df.mean()) # Replace the na with column mean (each item's mean)
        r_matrix = rating_df.values
        ratings_mean = np.mean(r_matrix, axis = 1)
        ratings_demeaned = r_matrix-ratings_mean.reshape(-1, 1)
        del rating_df,r_matrix

        U, sigma, Vt = svds(ratings_demeaned, self.__clusters)
        sigma = np.diag(sigma)
        all_predicted = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
        self.__preds_df = pd.DataFrame(all_predicted,index=users, columns = items)
        del ratings_mean, ratings_demeaned, all_predicted, sigma, U, Vt, users, items

    def __getBaseRatings(self):
        ratings = self.__ratings.iloc[:,0:3].copy()
        ratings.columns = ['user_id','item_id','rating']
        ratings['user_id'] = ratings['user_id'].apply(str)
        ratings['item_id'] = ratings['item_id'].apply(str)
        ratings.sort_values(['user_id','rating'],ascending=False,inplace=True)
        users = ratings.groupby('user_id').agg({'item_id': [lambda x: ','.join(x),'count']}).reset_index()
        users.columns = ['user_id','item_id','counts']
        users['counts'] = users['counts'].apply(lambda x: 10 if x>10 else x)
        ratings = ratings.pivot(index = 'user_id', columns ='item_id', values = 'rating')
        self.__rec_df = pd.DataFrame(data = ratings.isna(),index=ratings.index,columns=ratings.columns)
        self.__getRatingsPreds(ratings)
        self.__users = users
        del ratings   
    def __getSimilarItemDf(self):
        similarity_df = self.__item_preds
        mdf=pd.DataFrame()
        for item in similarity_df.index:
            sm = pd.DataFrame(similarity_df.loc[item].sort_values(ascending=False)[0:self.__top])
            try:
                sm = sm.drop(index=item).reset_index()
            except:
                sm = sm.reset_index()
            sm.columns = ['item_id','score']
            sm.index= [item]*(len(sm))
            mdf = pd.concat([mdf,sm])
        mdf['alg'] = 'content'
        return mdf
    def __getCFrecommends(self, user):
        similarity_df = self.__preds_df
        selection_df = self.__rec_df
        res = similarity_df.loc[[user],selection_df.loc[user,:]].loc[user].sort_values(ascending=False)[0:self.__top]
        pla_df = pd.DataFrame({'item_id':res.index,'score':res.values.round(2)})
        pla_df['alg']='user'
        return pla_df
    def __getAllRecommends(self):
        users = self.__users.iloc[:,0].values
        recommends = pd.DataFrame()
        if self.__noItem:
            itemSimilarityDf = self.__getSimilarItemDf()
        for user_id in users:
            use_df = self.__getCFrecommends(user=user_id)
            if self.__noItem:
                top_n = list(self.__users[self.__users['user_id']==user_id]['counts'])[0]
                hist_list = list(self.__users[self.__users['user_id']==user_id]['item_id'])[0].split(',')
                con_df = itemSimilarityDf.loc[hist_list[0:top_n]].drop_duplicates(['item_id'],keep='first')
                con_df = con_df[~con_df['item_id'].isin(hist_list)]
                mean_item_score = np.mean(con_df['score'])
                mean_user_score = np.mean(use_df['score'])
                con_df['score'] = con_df['score']*mean_user_score*self.__content_weightage
                use_df['score'] = use_df['score']*mean_item_score*self.__ratings_weightage
                userRecomDf = pd.concat([con_df,use_df])
                userRecomDf = userRecomDf.groupby('item_id')['score'].agg([np.mean]).sort_values('mean',ascending=False).iloc[0:self.__top].reset_index()
            else:
                userRecomDf = use_df
                userRecomDf = userRecomDf.groupby('item_id')['score'].agg([np.mean]).sort_values('mean',ascending=False).iloc[0:self.__top].reset_index()
            userRecomDf['User_id'] = user_id
            recommends = pd.concat([recommends,userRecomDf])
        return recommends
    def fit(self, ratings,item_df=pd.DataFrame()):
        self.__noItem = True
        self.__item_df = item_df
        self.__ratings = ratings
        if item_df.empty:
            self.__noItem = False
        else:
            self.__item_df.index = list(map(str, self.__item_df.index))
            self.__getItemPreds()
        self.__getBaseRatings()
        self.__recommends = self.__getAllRecommends()
    def predict(self,users=1):
        if users==1:
            return self.__recommends
        users = list(map(str, users))
        return self.__recommends[self.__recommends.User_id.isin(users)]
