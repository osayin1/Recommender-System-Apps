
# coding: utf-8

# In[18]:

import itertools
import ipdb
import pandas as pd
import numpy as np


# In[12]:

def rename_unknown_user(x):
    #x is a column (series of userID or userName values)  
    unknown_cnt = 0
    for idx in range(len(x)):
        val = x.iloc[idx]
        if val == 'unknown':
            x.iloc[idx] = val + str(unknown_cnt)
            unknown_cnt += 1
    
    if False:  #This turned out to be much slower 
        x_unknown = x.loc[x == 'unknown']       
        for idx in range(len(x_unknown)):
            #print idx
            x.iloc[x_unknown.index[idx]] = 'unknown' + str(idx)
    
    return(x) 


# In[13]:

def drop_duplicate_rating(df):
    # Drop all ratings but the most recent one for any particular user-movie pair. 
    # Generalize in the future to include options of using most helpful rating as well. 
    group_user_video = df.groupby(['userID','videoID'])

    drop_ind = []
    for idx, group in group_user_video:
        if len(group) > 1:
            #print idx
            #import debug
            group.sort(['review time'], inplace=True)
            drop_ind.extend(group.index[:len(group)-1])
            
    return df.drop(drop_ind,axis=0)


# In[3]:

def create_ratingsByUser_dict(df_grouped):
    dict_byUser = {}
    for uid, ratings_uid in df_grouped:
        ratings_uid.index = range(len(ratings_uid))
        dict_byUser[uid] = ratings_uid
        #import debug
    return dict_byUser    


# In[5]:

if False: #Older version.. 
    def compute_SlopeOne_dev(df):
        '''
        Weighted Slope One Prediction (Lemire et al. "Slope One Predictors for Online Rating-Based Collaborative Filtering")

        Compute item-item deviations in a dict. key := (movie_i,movie_j)
        '''

        users = df['userID'].unique()  #by userID

        devs = {}   #dict containing movie-to-movie rating deviations
        card = {}   #dict containing cardinalities for movie pairs
        user_movies = {} #dict containing the list of movies for each user

        for user in users:
            movies_user = df.loc[df['userID'] == user]
            #import debug
            user_movies[user] = list(movies_user['videoID'])
            #print 'user ' + user + 'rated ' + str(len(movies_user)) + ' movies'

            if len(movies_user) > 1:
                for pair in itertools.permutations(movies_user['videoID'],r=2):  #can change to combinations() for better memory usage 
                        #import debug 
                        dev_rating = movies_user['rating'].loc[movies_user['videoID'] == pair[0]].values - movies_user['rating'].loc[movies_user['videoID'] == pair[1]].values

                        if pair in devs:
                            devs[pair]  = devs[pair] + dev_rating
                            card[pair] += 1  
                        else:
                            devs[pair] = dev_rating        
                            card[pair] = 1   

                        #import debug
        for key in devs:
            devs[key] = devs[key] / float(card[key])   

        return (devs,card,user_movies)    


# In[4]:

def compute_SlopeOne_dev(df,user_movies):
    '''
    Weighted Slope One Prediction (Lemire et al. "Slope One Predictors for Online Rating-Based Collaborative Filtering")
    
    Compute item-item deviations in a dict. key := (movie_i,movie_j)
    '''
     
    devs = {}   #dict containing movie-to-movie rating deviations
    card = {}   #dict containing cardinalities for movie pairs
    
    
    for user in user_movies:
        movies_user = user_movies[user]
        #import debug
        #print 'user ' + user + 'rated ' + str(len(movies_user)) + ' movies'
        
        if len(movies_user) > 1:
            for pair in itertools.permutations(movies_user['videoID'],r=2):  #can change to combinations() for better memory usage 
                    #import debug 
                    dev_rating = movies_user['rating'].loc[movies_user['videoID'] == pair[0]].values - movies_user['rating'].loc[movies_user['videoID'] == pair[1]].values

                    if pair in devs:
                        devs[pair]  = devs[pair] + dev_rating
                        card[pair] += 1  
                    else:
                        devs[pair] = dev_rating        
                        card[pair] = 1   

                    #import debug
    for key in devs:
        devs[key] = devs[key] / float(card[key])   
        
    return (devs,card)    


# In[14]:




# In[15]:

def recommend_movie(pred_ratings,num_recommend):
    #recommend movies with the num_recommend predicted ratings for each user 
    pred_byUser = pred_ratings.groupby('userID')
    
    recom_dict = {}  #dict for recommendations per user. key:userID, value:list of recommended movieIDs

    for user, val in pred_byUser:
        # Compute #movies with a 5 rating. If more than num_recommend, randomly select num_recommend movies. 
        movies_5star = val['videoID'].loc[val['rating'] == 5.0]
        if len(movies_5star) > num_recommend: 
            recom_movies = movies_5star.iloc[np.random.randint(0,len(movies_5star),num_recommend)]
        else:
            sort_val = val.sort(['rating'], ascending=False)
            recom_movies = sort_val['videoID'].head(num_recommend)
        
        #import debug
        recom_dict[user] = pd.DataFrame(recom_movies,columns=['videoID'])
        
    return recom_dict


# In[20]:

def predict_ratings_slopeOne(df_rating,movies,slope1_dev,slope1_card):
        users_pred = df_rating['userID'].unique()  

        #df_pred = pd.DataFrame(columns=['rating','userID','videoID'])
        df_pred = []
        no_pred = 0 #num of new predictions

        for user in users_pred:
            df_rated     = df_rating.loc[df_rating['userID'] == user]
            rated_movies = df_rated['videoID']
            pred_movies  = list(set(movies).difference(set(rated_movies)))  #movies to predict ratings for
            #import debug

            for mov_j in pred_movies:
                sum_pred_scores = 0
                sum_card = 0 #sum of cardinalities

                for mov_i in rated_movies:
                    mov_pair = (mov_j,mov_i)    #tuples are hashable as keys
                    #import debug
                    if mov_pair in slope1_card:
                        rating_i = np.array(df_rated['rating'].loc[df_rated['videoID'] == mov_i]) #user's rating for mov_i
                        sum_card   += slope1_card[mov_pair]
                        sum_pred_scores += (slope1_dev[mov_pair] + rating_i) * float(slope1_card[mov_pair])

                if sum_card > 0:   #We have a prediction for user's rating of mov_j! 
                    no_pred += 1 
                    pred_rating = sum_pred_scores / float(sum_card) #user's predicted rating for mov_j
                    if pred_rating < 1:
                        pred_rating = 1.0
                    elif pred_rating > 5:
                        pred_rating = 5.0

                    #Add to dataFrame
                    #ind = pd.Series(np.arange(no_pred))
                    #new_row = pd.DataFrame(data = {'rating': pred_rating, 'userID': user, 'videoID': mov_j},index=ind.tail(1))
                    #df_pred = df_pred.append(new_row)

                    #import debug
                    df_pred.append([pred_rating,user,mov_j])
                    #import debug

        return pd.DataFrame(df_pred,columns=['rating','userID','videoID'])


# In[16]:

def update_slopeOne_devs(df,devs,card,user_movies):
    for idx_i, rating_row in df.iterrows():
        #import debug
        user_i   = rating_row.userID
        movie_i  = rating_row.videoID
        rating_i = rating_row.rating

        df_row = []
        df_row.append([rating_i,user_i,movie_i])
        df_row = pd.DataFrame(df_row,columns=['rating','userID','videoID'])
        #import debug
        if user_i not in user_movies: #new user
            print 'new user!'
            user_movies[user_i] = df_row
            #import debug
        else:
            user_ratings = user_movies[user_i]
            movies_rated = list(user_ratings['videoID'])
            #import debug
            if movie_i in movies_rated:
                print 'Already rated before..'
                #update user_movies_train
                #ignore this rating for now..
            else:
                print 'User has not rated this movie before..'
                for movie_j in movies_rated:
                    pair_ij = (movie_i,movie_j)
                    dev_ij = rating_i - user_ratings['rating'].loc[user_ratings['videoID'] == pair_ij[1]].values
                    if pair_ij in devs:
                        devs[pair_ij] = (devs[pair_ij]*float(card[pair_ij]) + dev_ij) / float(card[pair_ij]+1)
                        card[pair_ij] += 1
                    else:
                        devs[pair_ij]  = dev_ij
                        card[pair_ij] = 1

                    pair_ji = (movie_j,movie_i)
                    devs[pair_ji] = -1.0 * devs[pair_ij]
                    card[pair_ji] = card[pair_ij]

                user_movies[user_i] = user_movies[user_i].append(df_row,ignore_index=True)
            #import debug
    
    return (devs,card,user_movies) 
    
    
    


# In[ ]:



