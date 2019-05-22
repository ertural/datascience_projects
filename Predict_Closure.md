---
title: "Predict_Restaurant_Closure"
author: "Ertuğrul Ural"
date: "20 05 2019"
output:   
  html_document:
  keep_md: true
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Problem Setup 
Given the large number of restaurants and the total size of this industry, I decided to create a model that might help investors decide whether they should lend/invest at a particular restaurant based on the likelihood that it is going to fail in the future. I wanted to explore what types of qualities help determine whether a business will close down or not. The machine learning algorithms I used on this project were logistic regression, random forest, and a gradient boosting machine. I compared the accuracy, precision, and recall of these different approaches. Using these different models, I was able to examine a problem that many business owners deal with personally.

## Load Libraries
```{r, message=FALSE}
#Load pacman package and p_load function installs missing packages and loads all the packages given as input  
if (!require("pacman")) install.packages("pacman")

pacman::p_load("tidyverse",      #for data manipulation and graphs
               "lubridate",      #for date manipulation
               "ggplot2",        #for visualization
               "data.table",     #for efficiency of Importing Large CSV files
               "dplyr",          #for data wrangling
               "randomForest",   #randomForest
               "h2o",            #R interface for 'H2O', the scalable open source machine learning platform
               "sp",             #classes and methods for spatial data
               "leaflet",        #for interactive maps
               "leaflet.extras", #for interactive maps
               "geosphere"       #compute distances and related measures for (longitude/latitude) locations
)

```

The Yelp dataset is downloaded from https://www.yelp.com/dataset/challenge. In total, there are 5,261,668 user reviews, information on 174,567 business and 146,350 checkins and 1,326,100 information on users. I will focus on these 4 tables which are business, review, checkins and users. 

## Load Data
```{r, message=FALSE}
business <- fread("~/yelp_business.csv", header = T, sep = ',')
reviews <- fread("~/yelp_review.csv", header = T, sep = ',')
checkins <- fread("~/yelp_checkin.csv", header = T, sep = ',')
users <- fread("~/yelp_user.csv", header = T, sep = ',')

```

Through extensive exploratory data analysis, restaurants in Toronto have been studied. Using this insights obtained, I will predict whether a restaurant in Toronto will be successful or not. I am only interested in restaurants that are in Toronto for my further analysis. So let's remove all other records.

```{r}
toronto <- business %>%
  mutate(rest = ifelse(grepl('Restaurants',categories), 1, 0)) %>%
  filter(rest == 1, city == "Toronto")

#Only reviews related with Toronto Restaurants
reviews_toronto  <- reviews %>% 
  filter(business_id %in% toronto$business_id) 

#Only checkin info related with Toronto Restaurants
checkins_toronto  <- checkins %>% 
  filter(business_id %in% toronto$business_id) 

#Only checkin info related with Toronto Restaurants
users_toronto  <- users %>% 
  filter(user_id %in% reviews_toronto$user_id) 

```

## Feature Engineering

Feature engineering is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If feature engineering is done correctly, it increases the predictive power of machine learning algorithms by creating features from raw data that help facilitate the machine learning process.

### Chain Restaurants

Is the restaurant part of a chain? If the restaurant name appears more than once in the list then it is considered to be part of a chain. This includes national or local chains.The restaurants that are part of chains are more likely to remain open. This is not surprising as restaurant chains usually operate at a higher profit margin than individual restaurants.For these reasons, I added a chain restaurant flag variable(rest_chain) and the number of restaurants in that chain(rest_chain_count) to the data set.

```{r}
#Identifying chain restaurants
RestCounts <- toronto %>%
  select(name) %>%
  group_by(name) %>%
  mutate(rest_chain_count=n()) %>%
  arrange(desc(rest_chain_count)) %>%
  mutate(rest_chain = ifelse(rest_chain_count>1,1,0)) %>%
  distinct()

#merge year on year with business dataset
toronto <- left_join(toronto, RestCounts, by = 'name')

```

### Restaurant Clusters

High restaurant density might be correlated with higher closure rates. This is probably due to increased competition. It seems interesting to look at this feature in comparison to similar restaurant density (i.e. density of restaurants within 1 km distance) I created geographic clusters using coordinate data to make the model understand the impact of competition better. In order to succeed that I've used the hierarchical clustering technique.

The core logic behind this algorithm is that it starts by calculating the distance between every pair of points and store it in a distance matrix. Then the algorithm puts every point in its own cluster. Subsequently it starts merging the closest pairs of points based on the distances from the distance matrix and as a result the amount of clusters goes down by 1. Then it recomputes the distance between the new cluster and the old ones and stores them in a new distance matrix. Lastly it repeats steps 2 and 3 until all the clusters are merged into one single cluster. I set 500 meters as a distance threshold. At the end of the analysis, I obtained 406 clusters.

```{r}
#If we want to create Clusters based on lat long variables
x <- toronto[["latitude"]]
y <- toronto[["longitude"]]

# convert data to a SpatialPointsDataFrame object
xy <- SpatialPointsDataFrame(
  matrix(c(x,y), ncol=2), data.frame(ID=seq(1:length(x))),
  proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84"))

# use the distm function to generate a geodesic distance matrix in meters
mdist <- distm(xy)

# cluster all points using a hierarchical clustering approach
hc <- hclust(as.dist(mdist), method="complete")

# define the distance threshold, in this case 500m
d=500

# define clusters based on a tree "height" cutoff "d" and add them to the SpDataFrame
xy$clust <- cutree(hc, h=d)
Cluster <- (xy$clust)
max(xy$clust) #406 Clusters

#Append dataframes
toronto <- cbind(toronto, Cluster)
```

### Heat Map and Cluster Map of Restaurants

It seems clearly that in Toronto restaurants are mostly in downtowns. When you move further away from the city, number restuarants for which we have information reduces. It is not difficult to guess that competition in the city center is much higher.

On the second interactive map Restaurant clusters can be seen. We can see that clusters are concentrated in the city center in accordance with the distribution of restaurants.

```{r, echo=FALSE}
# Heat Map of the restaurants in Toronto
leaflet(toronto) %>%
  addTiles() %>%
  addHeatmap(lng=~longitude, lat=~latitude,
             blur = 20, max = 0.05, radius = 15)

# Interactive map for the clusters of the restaurants in Toronto
leaflet(data = toronto) %>% 
        addTiles() %>%
        addLabelOnlyMarkers(data=toronto,
                            lng=~longitude, lat=~latitude,
                            label=~as.character(Cluster),
                            clusterOptions = markerClusterOptions(),
                            labelOptions = labelOptions(noHide = T,
                                                  direction = 'auto'))

```

### USER AND REVIEW DATA

I've identified the dates that restaurants received their first and last review on the Yelp dataset.So I calculated the year on yelp variable for each restaurant. The restaurants that are older in the Yelp app may be less closed or vice versa.

```{r}
#Merge user and review data
reviews_users_toronto <- left_join(reviews_toronto, users_toronto, by = 'user_id',all.x = TRUE)

# calculate year of review
reviews_users_toronto$year <- as.numeric(substr(reviews_users_toronto$date,1,4))

# calculate latest year open
reviews_users_toronto$lyo <- ave(as.numeric(substr(reviews_users_toronto$date,1,4)),reviews_users_toronto$business_id, FUN = max)
 
# calculate first year open
reviews_users_toronto$fyo <- ave(as.numeric(substr(reviews_users_toronto$date,1,4)),reviews_users_toronto$business_id, FUN = min)

## Calculate years on yelp
yoy <- as.data.frame(group_by(reviews_users_toronto,business_id) %>%
                             summarize(lyo = mean(lyo, na.rm = TRUE),
                                       fyo = mean(fyo, na.rm = TRUE)))
 
#merge year on year with business dataset
toronto <- merge(toronto,yoy,by = 'business_id', all.x = TRUE)
toronto$yoy <- toronto$lyo - toronto$fyo + 1

#removing lyo and fyo variables
toronto$lyo <- NULL
toronto$fyo <- NULL
```

In order to understand the review characteristics better, I have calculated the averages of useful, fun and cool reviews for each restaurant using the user and review data set. I have calculated the average star value for restaurants using the ratings given by the users.

```{r}
## Calculate review characteristics by business and merge
review_char <- as.data.frame(group_by(reviews_users_toronto, business_id) %>%
                             summarize(bus_mean_useful = mean(useful.x, na.rm = TRUE),
                                       bus_mean_funny = mean(funny.x, na.rm = TRUE),
                                       bus_mean_cool = mean(cool.x, na.rm = TRUE)))
 
## Merge
toronto <- merge(toronto,review_char,by = 'business_id', all.x = TRUE)

## Calculate different characteristics of the review distribution for each business
stars_char <- as.data.frame(group_by(reviews_users_toronto,business_id) %>%
                             summarize(bus_mean_star = mean(stars, na.rm = TRUE)))

# merge
toronto <- merge(toronto,stars_char,by = 'business_id', all.x = TRUE)
```

I normalized the number of checkins and number of reviews according to the year spent in the yelp app. It seems that some restaurants have been active in this application for many years, some of them not. So, it is important to normalize the these ratios to reduce the bias.

```{r}
#Total number of checkins
total_checkins <- checkins_toronto %>% 
  group_by(business_id) %>%
  mutate(total_checkins = (sum(checkins))) %>%
  select(business_id, total_checkins)

#Merge
toronto <- merge(toronto,total_checkins,by = 'business_id', all.x = TRUE)

#Normalize
toronto <- toronto %>%
  mutate(norm_checkins = floor(total_checkins/yoy)) %>%
  mutate(norm_review_counts = floor(review_count/yoy))

```

I've added clusters to include competition as a latent factor in the model. After creating the clusters, I calculated the average star, the average number of reviews, the average year spent in the application and the average number of checkins for the restaurants in each cluster. In this way, I have achieved ratios. Through these ratios I analyzed the relative success of the restaurants in a cluster.

```{r}
#Creating Cluster based features in order to compare restaurants with their peers
cluster_avg <- toronto %>%
            group_by(Cluster) %>% 
            mutate(number_of_rest = n(),
              cluster_avg_stars = mean(stars, na.rm = TRUE),
              cluster_avg_review_count = mean(review_count, na.rm = TRUE),
              cluster_avg_yoy = mean(yoy, na.rm = TRUE),
              cluster_avg_total_checkins = mean(total_checkins, na.rm = TRUE)) %>%
           select(Cluster,number_of_rest, cluster_avg_stars, cluster_avg_review_count, cluster_avg_yoy, 
                  cluster_avg_total_checkins) %>%
           distinct() %>%
           arrange(desc(number_of_rest)) 

#Merging two datasets
toronto2 <- merge(toronto, cluster_avg, by= c("Cluster"), all.x = TRUE, allow.cartesian=FALSE)

#Creating Ratio features in order to compare dynamic features of the hotels in a better way
toronto3 <- toronto2 %>% mutate( 
              ratio_review_count = (review_count/cluster_avg_review_count),
              ratio_stars = (stars/cluster_avg_stars),
              ratio_avg_yoy = (yoy/cluster_avg_yoy),
              ratio_avg_checkins = (total_checkins/cluster_avg_total_checkins))

#Impute NA's with 0
toronto3[is.na(toronto3)] <- 0
```

I've converted the is_open variable to is_closed. The target variable thus received positive values.

```{r}
## Make a closed business the 'positive' case
table(is.na(toronto3$is_open))
toronto3$closed <- as.factor(ifelse(toronto3$is_open == 0,1,0))
toronto3 <- select(toronto3,-is_open)
summary(toronto3$closed)
```

What I was most curious about was whether there was a relationship between the stars given to restaurants and the closure of restaurants. According to the chart below, it is difficult to say that we there's such a relationship.

```{r}
ggplot(toronto3, aes(fill = closed)) + geom_histogram(aes(x = stars,y=..density..),
                                                       position = 'identity',
                                                binwidth = 0.5, alpha = 0.7) +
  scale_fill_manual(values = c('orange','blue')) +
  theme_bw() + xlab('Star Rating') + ylab('Density') +
  ggtitle('Star Rating Histograms', subtitle = 'for open and closed businesses')
```

## Modeling

I used the h2o package for modeling. R interface for 'H2O', is the scalable open source machine learning platform that offers parallelized implementations of many supervised and unsupervised machine learning algorithms such as Generalized Linear Models, Gradient Boosting Machines (including XGBoost), Random Forests, Deep Neural Networks.

As a result of the experiments I did before, I decided not to include some variables in the model.I splitted the data set into three groups: train, validation and test. I wanted to understand in terms of the Bias variance trade-off if there would be a significant difference between the model's success in the test set and train set.

```{r, message=FALSE}
## DROP CERTAIN VARIABLES ##
cols <- c("Cluster","business_id", "name", "neighborhood", "address", "state", "city", "postal_code", "categories", "rest", "total_checkins", 
          "review_count", "latitude", "longitude", "stars", "ratio_stars", "ratio_avg_checkins")

#create new dataset
toronto4 <- toronto3[ , !(names(toronto3) %in% cols)]

library(h2o)        # Professional grade ML pkg

# Initialize H2O JVM
h2o.init()

# Split data into Train/Validation/Test Sets
toronto4_data_h2o <- as.h2o(toronto4)

# create train, validation, and test splits
split_h2o <- h2o.splitFrame(toronto4_data_h2o, c(0.7, 0.15), seed = 13)

train_h2o <- h2o.assign(split_h2o[[1]], "train" ) # 70%
valid_h2o <- h2o.assign(split_h2o[[2]], "valid" ) # 15%
test_h2o  <- h2o.assign(split_h2o[[3]], "test" )  # 15%

# variable names for resonse & features
y <- "closed"
x <- setdiff(names(train_h2o), y) 

```

Ridge and Lasso regression are some of the simple techniques to reduce model complexity and prevent over-fitting which may result from simple linear regression.The only difference between them is instead of taking the square of the coefficients, magnitudes are taken into account. This type of regularization (L1) can lead to zero coefficients i.e. some of the features are completely neglected for the evaluation of output. So Lasso regression not only helps in reducing over-fitting but it can help us in feature selection. First I used the Lasso algorithm to understand which attributes are more significant for the model.

```{r}
#----------------------------------------------------------------------------------------------------------------------------------------------
#LASSO(AUC Train = 0.7584122, AUC=0.7617468, Acc Test = 0.6414048, Recall Test = 0.8304094, Precision Test = 0.4625407) 
#----------------------------------------------------------------------------------------------------------------------------------------------
fit_lasso <- h2o.glm(x = x,
                    y = y,
                    training_frame = train_h2o,
                    validation_frame = valid_h2o,
                    standardize = TRUE,
                    alpha =  1,
                    lambda_search = TRUE,
                    family = "binomial",
                    model_id = "fit_lasso",
                    seed = 13)

#Summary of the model
h2o.auc(fit_lasso, train = TRUE)
h2o.auc(fit_lasso, valid = TRUE)

# Predict on hold-out set, test_h2o
pred_h2o <- h2o.predict(object = fit_lasso, newdata = test_h2o)

# Prep for performance assessment
test_performance <- test_h2o %>%
    tibble::as_tibble() %>%
    select(closed) %>%
    add_column(pred = as.vector(pred_h2o$predict)) %>%
    mutate_if(is.character, as.factor)

# Confusion table counts
confusion_matrix <- test_performance %>%
    table() 
confusion_matrix

```

It is important to understand is that the accuracy can be misleading: 65% does not sound pretty good especially for modeling, also if we just pick closed = yes we would get an accuracy of about 30%. Does sound better now.

Before we make our final judgement, let’s dive a little deeper into precision and recall. Precision is when the model predicts yes, how often is it actually yes. Recall (also true positive rate or specificity) is when the actual value is yes how often did the model catch the true positive results. 

Most investors would probably prefer to incorrectly classify restaurants not closing instead of missing the closed ones. Because it’s important to not miss at risk money, and that's why they will really care about recall more or when the actual value is closed = YES how often the model predicts YES. Recall for our model is 83,04%. In an lender context, this is 82% of unsuccessful restaurants could potentially be targeted prior to funding. On the other hand, the precision of closed restaurants is very poor but can be further improved but there is always a trade-off with the precision of closed restaurants and recall of closed restaurants.

```{r}
# Performance analysis
tn <- confusion_matrix[1]
tp <- confusion_matrix[4]
fp <- confusion_matrix[3]
fn <- confusion_matrix[2]

#Accuracy, recall and precision
accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)

#Show metrics
tibble(
    accuracy,
    recall,
    precision)

```

## Feature Selection & Visualization

The feature importance that resulted from this model is shown below. The features that contributed towards the restaurants closures are shown in blue, while the features that contributed towards the restaurants remain open are shown in orange.

The normalized review count (i.e. the number of reviews normalized according to years company spent in the application) is the most important feature that contributes towards the restaurant remaining open. It is hard to strictly label this metric as an indication or a cause of success. A large number of reviews is an indication of higher traffic in restaurants but it is also a reason to appear higher in Yelp search results, which by itself can drive more traffic.

The second most important feature, as ranked by our model, is whether the restaurant is part of a chain and the total number of restaurants at that particular chain(shows the strength of the chain). The restaurants that are part of larger chains are more likely to remain open.

Restaurants spent more many years on Yelp are more likely to remain open. The Yelp business is a business where the owner has put the effort to go on Yelp and declare the business as their own. In that sense, a positive correlation with restaurant success was expected.

High restaurant density is correlated with higher closure rates. This is probably due to increased competition. It is interesting to look at cluster average star feature. Large number of restaurants having higher rating in a specific cluster might be negative for ohter restaurants in terms of competition. This says that, for instance, owning a Chinese restaurant in an area with a large number of restaurants having higher rating is generally negative for this Chinese restaurant as expected. On the other hand owning a a Chinese restaurant in an area with large number of restaurants spent less year on Yelp is generally positive for that restaurant this time. 

```{r}
#compute variable importance and performance
h2o.varimp_plot(fit_lasso, num_of_features = 10)

## Feature Importance
features_lasso <- as.data.frame(h2o.varimp(fit_lasso))
```

In the graphs below, we can see how the closed restaurants are showing the density according to the important variables. We can say that restaurants having less than 10 normalized number of reviews are more likely to close. In addition, we can say that the restaurants which are active in the Yelp applicationless than 5 years is more likely to close than the other restaurants.

```{r}
#Density plot of closed ~ norm_review_counts
ggplot(data = toronto4, aes(norm_review_counts)) + 
  geom_density(alpha=0.4,aes(fill = as.factor(closed))) +
  xlim(0, 50) 

#Density plot of closed ~ rest_chain_count
ggplot(data = toronto4, aes(rest_chain_count)) + 
  geom_density(alpha=0.4,aes(fill = as.factor(closed))) +
  xlim(0, 10) 

#Density plot of closed ~ yoy
ggplot(data = toronto4, aes(yoy)) + 
  geom_density(alpha=0.4,aes(fill = as.factor(closed))) +
  xlim(0, 10) 

#Density plot of closed ~ yoy
ggplot(data = toronto4, aes(cluster_avg_review_count)) + 
  geom_density(alpha=0.4,aes(fill = as.factor(closed)))

```

Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. The guiding heuristic is that good predictive results can be obtained through increasingly refined approximations. H2O’s GBM sequentially builds regression trees on all the features of the dataset in a fully distributed way - each tree is built in parallel.

Since boosted trees are derived by optimizing an objective function, basically GBM can be used to solve almost all objective function that we can write gradient out.

GBMs are more sensitive to overfitting if the data is noisy. We can see that the AUC value, which is 0.89 in the train data, has decreased drastically to 0.79 in the validation data. This reduction is above 10% and it's hard to claim that this is a very reliable model. To reduce complexity, we can change default parameters like the minimum number of samples (or observations) which are required in a node to be considered for splitting, the maximum depth of a tree or the number of features to consider while searching for a best split

```{r}
#----------------------------------------------------------------------------------------------------------------------------------------------
#GBM(AUC Train = 0.8947347, AUC Valid =0.7910086, Acc Test = 0.7116451, Recall Test = 0.7222222, Precision Test = 0.5323276) 
#----------------------------------------------------------------------------------------------------------------------------------------------
fit_gbm1 <- h2o.gbm(x = x,
                    y = y,
                    training_frame = train_h2o,
                    validation_frame = valid_h2o,
                    model_id = "fit_gbm1",
                    seed = 13)

#Summary of the model
h2o.auc(fit_gbm1, train = TRUE)
h2o.auc(fit_gbm1, valid = TRUE)

# Predict on hold-out set, test_h2o
pred_h2o <- h2o.predict(object = fit_gbm1, newdata = test_h2o)

# Prep for performance assessment
test_performance <- test_h2o %>%
    tibble::as_tibble() %>%
    select(closed) %>%
    add_column(pred = as.vector(pred_h2o$predict)) %>%
    mutate_if(is.character, as.factor)

# Confusion table counts
confusion_matrix <- test_performance %>%
    table() 
confusion_matrix

```

When we examine the confusion matrix created from the predictions out of the test data set, we see that the total accuracy and the precision value increase and the value of the recall decreases.

```{r}
# Performance analysis
tn <- confusion_matrix[1]
tp <- confusion_matrix[4]
fp <- confusion_matrix[3]
fn <- confusion_matrix[2]

#Accuracy, recall and precision
accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)

#Show metrics
tibble(
    accuracy,
    recall,
    precision)

```

Distributed Random Forest (DRF) is a powerful classification and regression tool. When given a set of data, DRF generates a forest of classification or regression trees, rather than a single classification or regression tree. Each of these trees is a weak learner built on a subset of rows and columns. More trees will reduce the variance. Both classification and regression take the average prediction over all of their trees to make a final prediction, whether predicting for a class or numeric value. 

The main limitation of the Random Forests algorithm is that a large number of trees may make the algorithm slow for real-time prediction. For data including categorical variables with different number of levels, random forests are biased in favor of those attributes with more levels.RF are harder to overfit than GBM. We can see that there isn't significant difference in AUC values between train and test data set. More reliable model than GBM.

```{r}
#----------------------------------------------------------------------------------------------------------------------------------------------
#RF(AUC Train = 0.7561989, AUC Valid = 0.7875165, Acc Test =0.7430684, Recall Test = 0.6081871, Precision Test = 0.5909091) 
#----------------------------------------------------------------------------------------------------------------------------------------------
fit_rf1 <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train_h2o,
                            validation_frame = valid_h2o,
                            model_id = "fit_rf1",
                            seed = 13)

#Summary of the model
h2o.auc(fit_rf1, train = TRUE)
h2o.auc(fit_rf1, valid = TRUE)

# Predict on hold-out set, test_h2o
pred_h2o <- h2o.predict(object = fit_rf1, newdata = test_h2o)

# Prep for performance assessment
test_performance <- test_h2o %>%
    tibble::as_tibble() %>%
    select(closed) %>%
    add_column(pred = as.vector(pred_h2o$predict)) %>%
    mutate_if(is.character, as.factor)

# Confusion table counts
confusion_matrix <- test_performance %>%
    table() 
confusion_matrix
```

When compared to the Lasso model, we can see that the confusion matrix result are more balanced than Lasso. However, if we think that the recall value is more critical for investors in this problem, it can be stated that the Lasso model is more successful than other models. However, investors will also pay attention to the precision value. RF can help us at this point. Perhaps an ensemble model can be developed with RF and Lasso. 

```{r}
# Confusion table counts
confusion_matrix <- test_performance %>%
    table() 
confusion_matrix

# Performance analysis
tn <- confusion_matrix[1]
tp <- confusion_matrix[4]
fp <- confusion_matrix[3]
fn <- confusion_matrix[2]

#Accuracy, recall and precision
accuracy <- (tp + tn) / (tp + tn + fp + fn)
recall <- tp / (tp + fn)
precision <- tp / (tp + fp)

#Show metrics
tibble(
    accuracy,
    recall,
    precision)
```


```{r}
## Feature Importance
h2o.varimp(fit_rf1)
```

## Summary

* This model was built for restaurants in Toronto for lending purposes and identifies restaurants that closed with a recall of 83%.

* Some very predictive features of this model were built using Yelp review, location and user metadata. These data sets helped me to construct relative metrics like restaurant density and quantities that are relative to surrounding restaurants.

* One lesson learned is that the most important factor that defines whether a restaurant will remain open is whether it is part of a chain. Restaurants that belong to chains close less frequently.

* Another lesson learned is that building a restaurant in an area with a lot of other restaurants is generally negative.

* One possible reason for a restaurant closure might be related with health inspection. Adding health inspection ratings as a feature in our model could increase its precision. Another reason for restaurant closure might be high rents. Adding rent pricing per region could help explain more restaurant closures.

* A change in population demographics in certain areas of a city can increase or decrease traffic to some restaurants. In order to quantify and understand sales potential of each restorant, we should first benefit from the socio demographic data. Through data made available by city administrations and Machine Learning algorithms we can overcome that problem.  


