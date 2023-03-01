# Movie Reccomender System
Mark Pop 
1. An Introduction to the Project

Recommendation systems have been a growing issue in the past twenty or so years due to the rising popularity of the internet. This is especially true in areas such as ecommerce where online stores suggest related items to customers and streaming services where previously unseen movies are shown that match the preference of the viewer. What techniques to use has been a big topic of discussion as of late. The issue is so important that Netflix, a popular movie streaming service, held a competition where the contestants would try to create a recommendation system that would outperform their current system by at least 10%, and the winner would receive one million dollars as a prize. “Participating teams submit predicted ratings for a test set of approximately 3 million ratings, and Netflix calculates a root-mean-square error (RMSE) based on the held-out truth. The first team that can improve on the Netflix algorithm’s RMSE performance by 10 percent or more wins a $1 million prize [1].” The team that had the best results in the competition implemented a technique called Matrix Factorization. I implemented this model with a data set of about one hundred thousand ratings and tested the efficiency (time to train model) and accuracy (precision and recall).
 
2. An Explanation of the Solution

“Matrix factorization is a way to generate latent features when multiplying two different kinds of entities.Collaborative filtering is the application of matrix factorization to identify the relationship between items’ and users’ entities [2].” In essence Matrix Factorization takes a matrix of ratings where users are the rows, movies are the columns, and ratings are the content. This is a very sparse matrix since each user only rates a hand full of movies. We find latent features to describe the relationship between user preference and movie characteristic. So, a feature might be that a movie has the genre of Action. If a user Brian likes Action movies, he is more likely to rate the movie highly. We decompose the original ratings matrix into two smaller matrices, P which is comprised of users for rows and features for columns, and Q which is comprised of features for rows and movies for columns. Once trained we can perform a dot product to create a matrix like the original ratings matrix. We use this new ratings matrix to recommend previously unseen movies to users.

The training process starts by creating the P and Q matrices initialized with random values. We loop through every single element in the original rating matrix that has a value, skipping the elements that are valueless, subtracting the corresponding P dot Q value at that position to find the error rate.             

e = (r<sub>ij</sub> - p<sub>i</sub> * q<sub>j</sub>)

Error is used to update the model through stochastic gradient descent. The model takes the original P and Q values at points i and j (p<sub>i</sub>, q<sub>j</sub>), calculates the error, and uses a derivation equation to update the p<sub>i</sub> and q<sub>j</sub> values at that step. 

p<sub>i</sub> <-- p<sub>i</sub> + &alpha;(e * q<sub>j</sub> - &beta; * p<sub>i</sub>)

q<sub>j</sub> <-- q<sub>j</sub> + &alpha;(e * p<sub>i</sub> - &beta; * q<sub>j</sub>)

Here we take the error multiplied by the  value, subtracting  multiplied by a beta  value to normalize the result. We multiply alpha, which is the step size, by the resulting value and add this to the original value to get our updated  value. We do the exact same thing with except we multiply the error by and the beta value by. An Objective Function is chosen to evaluate how close the PQ matrix is to the original ratings matrix. “One intuitive object unction is the squared distance. To do this, minimize the sum of squared errors over all pairs of observed entries [3]:”

&Sigma;<sub>ij</sub>(r<sub>ij</sub> - p<sub>i</sub> * q<sub>j</sub>)<sup>2</sup>

This process is repeated until we’ve minimized the error, incrementally optimizing the P and Q matrices until a specified number of steps is reached or an error threshold is crossed.

3. Description of the Results

The metrics we used totest the model were efficiency (time to train the model), and accuracy (precision and recall). Efficiency for our Matrix Factorization model initially was about 16 hours to train without any of the mean or bias values calculated. We believe this was due to several reasons. One was the resources allocated to the machine training the model, about 4 GB of RAM. The second was the computations needed to train the model. We set the number of latent features to 75 for minimal error, but the downside was the increased time to train. Precision for the model which we defined as: 

Precision@k = R<sub>i</sub>/Nr

Here we define  as the number recommended movies that are also relevant, and  as the number of recommended movies. By relevant we mean ratings over a certain threshold, here we used 3.5. Recall for the model is defined as: 

Precision@k = R<sub>i</sub>/Tr

Where  again is the number of recommended movies that are also relevant, and  is the total number of relevant items. In our initial testing for Matrix Factorization, we got a Precision score of 0.51 and a Recall score of 0.42. 

There were many attempts at producing the lowest error by finding the optimal alpha (learning rate), beta (normalization value), and feature values. We started at an alpha value of 0.01, a beta of 0.02, and 3 latent features. At this stage the model would reach an error of about 50k-65k at around step 125, then the error rate would increase with each successive step. This indicates to me that the model reached some sort of local minima. We decided to increase the number of features to 20 keeping the other parameters (alpha & beta) the same. This yielded a resulting error value of 16k at around 400 steps. We attempted to vary the alpha value at this stage, but it only produced worse results:

- Alpha = 0.003: Error of 16k at step 280
- Alpha = 0.002: Error of 18k at step 230
- Alpha = 0.001: Error of 24k (step not recorded)

Number of latentfeatures seemed to be the crux of the problem, so we continued to try differentfeature values. The next step was a feature value of 50, where alpha was kept at 0.01 and beta was kept at 0.02, resulting in an error of about 11k at step 85. Finally, we tried 75 feature value with the same alpha and beta values, and it produced a result error of 2.8k at 600 steps. Since the steps to calculate error were not consistent, we decided to analyze the model at 5 different key feature values (5, 10, 20, 50, and 75) at 25 steps to get a consistent measurement. Latent feature value of 75 was used to calculate the previous model because it had the lowest error rate as shown in the graph below.  

"In any cross-validation we split the data such as some of it is being fitted on, and the rest of the data is used for testing [4]." Here we used fourfold cross validation to train and test the model, taking 25% of the data and setting it aside for testing and using the remaining 75% for training, repeating this for every quadrant of the main matrix. Essentially there were four models trained so we took each 25/75 pair and calculated the error rate for each one, taking the average error for all four. The error for the model was 86,204,049 when it was running initially with 75 features run at 750 steps.

4. Extra Work

I decided to take things a step further for this portfolio project analysis. The P and Q matrices describe the interaction between user and item, but I feel as though this is inadequate. Yehuda Koren, et al. explains this point in their paper well when they state, “much of the observed variation in rating values is due to effects associated with either users or items, known as biases or intercepts, independent of any interactions. For example, typical collaborative filtering data exhibits large systematic tendencies for some users to give higher ratings than others, and for some items to receive higher ratings than others. After all, some products are widely perceived as better (or worse) than others [1].” The suggestion here is to add an overall mean value for all ratings as a base, then add bias values for both users and movies. These bias values would be the average rating for the user or movie minus the global average rating. The new error:

e = (r<sub>ij</sub> - &mu - b<sub>i</sub> - b<sub>j</sub> - p<sub>i</sub> * q<sub>j</sub>)

Here  is the global mean of all ratings in the rating matrix,  is the bias value for that specific user,  is the bias value for that specific movie. We use this updated error calculation to update the  and  values so that when we create the new matrix, the basis is now the mean and the biases. This will mean that  and  will account for less of the model and lead to more accurate predictions for previously unseen movies. The Objective Function is now updated as well: 

&Sigma;<sub>ij</sub>(r<sub>ij</sub> - &mu - b<sub>i</sub> - b<sub>j</sub> - p<sub>i</sub> * q<sub>j</sub>)<sup>2</sup>

In the previous graph we see those 50 features and 75 features are only marginally different after 25 steps. I decided to again analyze the 50 and 75 feature values to see if after 100 steps the difference in error increased at all. The reason behind this was because the 75-feature value takes longer to train the model than the 50-feature value. If the difference in error is still only marginal after four times the steps, then I would use the 50-feature value to train the model on this iteration. Efficiency was one area where this model performed the worst so any means of cutting it without drastically reducing precision will be taken. 

As we can see the 50 and 70 feature values are practically indistinguishable at 100 steps. Therefore the 50 was used in this iteration to save some time on training.

Rebuilding the model with 50 features at 500 steps with both the mean and the biases included yielded an error rate of 96,211,185. The original expectation was that this new model would perform better in terms of error. My guess is that the lower feature value and smaller step size contributed to the higher error rate. The new model got a Precision score of 0.66 and a Recall score of 0.54, both better than the first iteration of the model, which had a Precision score of 0.51 and a Recall score of 0.42. In terms of efficiency the new model completed training in about 6 hours which was substantially faster than the original iteration of the model which ran for 16 hours to fully train all four folds. I believe this improvement is due to both the change in feature value and step size, as well as adding 8 GB additional RAM to the machine the model was trained on.

4. New Skills

This program has helped me grow by leaps and bounds in understanding machine learning principles such as how to better research the topic, test the model and analyze the results, and manage extremely large data sets. One new skill in exploring several different articles and research papers, and how to learn from those sources better. I got to take what others have done and implement the relative pieces to my project to help improve it. 

How to properly test the model and analyze results were some other big learning points for me in this project. When trying to find the best alpha, beta, and feature numbers my parameters varied too much, specifically step size was inconsistent. I kept alpha and beta the same while trying to find an optimal feature value, but the step size for each test was different. I learned to keep everything consistent and only change one parameter at a time when testing. I believe that the low results were because such a large portion of the data was removed for testing, drastically reducing the accuracy of the model. If I had more time, I would implement ten-fold cross validation instead of the fourfold cross validation to build the model using 10% for testing and the rest for training. I believe this would greatly improve the results since the original ratings matrix is so sparse and using 90% of the data to train would help mitigate that. 

Since the data set was so large, around one hundred thousand values, training the model took an extremely long time. There were several places where I had to investigate how to make the model run more efficiently, such as looking at the machine that was training the model. One bottleneck I came across was that there was only 4 GB of RAM allocated when I initially performed the training which led to a 16-hour execution of the program. In my previous projects the data had been so small that optimizing for time was never an issue, here it was one of the main problems. I ended up increasing the RAM to 12 GB which drastically reduced the amount of time it took to train. I also performed some analysis on the model and found that the difference in error between the 50 feature values and 75 feature values were almost negligible so to save time I trained the second iteration of the model on the 50-feature value. 

5. References 

[1] Google. (2020, Feb. 10). Matrix Factorization [Online]. Available:
https://developers.google.com/machine-learning/recommendation/collaborative/matrix.

[2] Chen D. (2020, Jul 8). Recommender Systems – Matrix Factorization [Online].
Available: Recommender
System — Matrix Factorization | by Denise Chen | Towards Data Science

[3] Koran Y. et al. (2009, Aug 07). "Matrix Factorization Techniques
for Recommender Systems." Computer [Online]. Vol. 42. Issue 8.
Available: https://www.inf.unibz.it/~ricci/ISR/papers/ieeecomputer.pdf

[4] Telavivi I. (2020, Jan 2). How to Use Cross-Validation for Matrix
Completion [Online]. Available: https://towardsdatascience.com/how-to-use-cross-validation-for-matrix-completion-2b14103d2c4c
