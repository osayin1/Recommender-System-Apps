# Recommender-System-Apps

This repository contains a colloborative filtering implementation on a large set of Amazon Instant Video review data (100,000 reviews). The data is available on the SNAP Website of Stanford University. (McAuley et al., "Hidden factors and hidden topics: understanding rating dimensions with review text", Proc of the 7th ACM conference on Recommender Systems, Oct 2013.)

The current implementaion utilizes a Weighted Slope One Predictor (Lamire et al., "Slope One Predictors for Online Rating-Based Collaborative Filtering", SIAM Data Mining, Apr 2005.). "recommender.py" contains a variety of functions for tasks such as computing item deviations, predicting movie ratings recommending movies to users. "Recommend_Amazon_Movies" is the iPython notebook that carries out the initial handling and preprocessing of the data and also executes the recommender.py functions to predict ratings and make movie recommendations. The mean absolute error of the predicted ratings on a validation set was 0.74 with ratings ranging between 1 and 5. Updating the predicted ratings and recommendations based on new reviews is also a handy feature of this implementation. 
