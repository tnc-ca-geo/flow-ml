# Predict natural flows using machine learning

## Introduction
Water is essential for California’s people, economy, and environment. Centuries of water management through dams and diversion have altered the flows in many streams and rivers, which can harm the freshwater ecosystems. The Nature Conservancy (TNC) and interns from University of San Francisco (USF) have written code to generate estimates of natural flows (expected streamflow in the absence of human modification) for streams and rivers in California.  The code uses machine learning techniques and is based on methods developed by the US Geological Survey (USGS).

Background Resources:

* California Natural Flows Database:  https://rivers.codefornature.org
* Zimmerman, JKH, Carlisle, DM, May, JT, et al. Patterns and magnitude of flow alteration in California, USA. Freshwater Biol. 2018; 63: 859– 873. https://doi.org/10.1111/fwb.13058 
* Stream flow data update pipeline code:  https://github.com/tnc-ca-geo/flow-update 

## Methods

**1. Initial Data Cleaning and Preparation**

1.1 The input data on stream flow, climate, and watershed conditions were assembled by TNC using the code in the flow-update repo listed above.  The data for just reference quality stream gages (places with minimal human modification to flow) are linked below.  For more information on the data source, see Zimmerman et al, 2018.

[1.1.a] (most recent) Training input data (Jan 1950 - Dec 2015 ) pulled from: https://tnc.box.com/s/rp5f24n6p82gcly2bv71wrb3de0o4djt
Earlier versions of training input are all located in this folder: https://tnc.box.com/s/e7w4tihokr6o28r49o8tuzckpy4jhznc<br /> 
[1.1.b] (most recent) Testing input data (Nov 2021 - Apr 2022) pulled from: https://tnc.box.com/s/y9s8x19smxnmztu5ctqguwjkma4udcqi
Earlier versions of testing input are all located in this folder:  https://tnc.box.com/s/n6j3u0afiu4u4e11racq9ppcjv991n0b

1.2. Columns with missing records and non-predictive independent variables('Unnamed: 0', 'year', 'gage_id', 'statistic', 'variable', 'NHDPLUSREG', 'comidyear', 'ECO3', 'AggEcoRegion') were dropped.
Reason why below variables are non-predictive:
 * ‘year’: natural flows were affected by monthly weather and not by year
 * ‘gage_id’: similar to non-repetitive indexing hence it provides no predictive insight
 * ‘statistic’: it will be either “mean”, “min” or ”max” which remains the same within each respective dataset 
 * ‘variable’: unchanged data filled with “observed” for every cell
 * ‘NHDPLUSREG’: this is the region label, eg: 16 for Great Basin, 17 for Pacific Northwest, 18 for California, however, we do not want the natural flows predicted to be segregated by regions. 
 * ‘comidyear’: a combination of unique flows comid and year, similar to ‘gage_id’, non-repetitive indexing hence it provides no predictive insight
 * ‘ECO3’: references one type of ecosystem region used by USGS but we are using ‘NewEco3’ column instead
 * ‘AggEcoRegion’: again we do not want the natural flows predicted to be segregated by aggregated eco regions but we want them to be segregated by ’NewEco3’ regions

1.3. Training and testing on dataset:<br /> 
In machine learning, a model's ability to generalize is central to the success of a model - meaning that the model is able to adapt to new and previously unseen data to make equally accurate predictions as compared to the predictions obtained during the training data. Ideally, we should not test on training data, because the model has already seen the data before, the prediction will fit closely and represent the training dataset a little too accurately. However, this good accuracy in the training dataset cannot guarantee the success of the model on new unseen data and this overfitting occurance will most likely cause the model to fail on new data predictions.

Hence, we recommend keeping training data separate from the testing data, and since we have a large enough dataset, we used a 80:20 training to testing split (common practice) for our dataset where on each run of the notebook, the split is randomized. The randomization works as a random sampling without replacement, to further prove the generalization ability of the model working with different data.

1.4. We used label encoding on Column ['NewEco3'],  to identify the different regions(Xeric, Coastal Mountains and Interior Mountains) because previous research by the USGS indicated differing flow relationships based on geographic region.

**2. Random Forest Modeling**
2.1 Hyperparameter Tuning<br /> 
Hyperparameter is a parameter whereby the value is used to control the learning process of machine learning, which in this case some of the main hyperparameter values are the n_estimators, max_depth, min_samples_leaf, max_features which controls the learning rate of our random forest machine learning model. When we make use of the RandomForestRegressor from the sclearn package (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), they have the hyperparameters set at certain default values, which we could change if we decided that there are better values that improves the learning rate and result of the model. The following are detailed descriptions of each hyperparameter.

Hyperparameters in RandomForestRegressor:<br /> 
n_estimators = number of trees in the forest<br /> 
max_depth = maximum number of levels in each decision tree<br /> 
min_samples_leaf = minimum number of data points allowed in a leaf node (leaf is the end node of a tree)<br /> 
max_features = maximum number of features considered before splitting a node/individual tree<br /> 
*Few built-in options available in Python to assign maximum features: 
 - Auto/None : This will simply take all the features which make sense in every tree.Here we simply do not put any restrictions on the individual tree.
 - sqrt : This option will take square root of the total number of features in individual runs. For instance, if the total number of variables are 100, we can only take 10 of them in an individual tree.”log2″ is another similar type of option for max_features.
 - 0.2 : This option allows the random forest to take 20% of variables in individual runs. We can assign and value in a format “0.x” where we want x% of features to be considered.

Rather than go with the default hyperparameters, we tested several random combinations of hyperparameters, ran the random forest model, and chose the set with the highest R-squared value.  The code allows you to set the number of random combinations to test. Within the hyperparameter tuning (hyper_parameter_search() function), the tuning function choses the hyperparameters and its R2 from each run of the RandomForestRegressor and the hyperparameter_csv_append() function records the values of these hyperparameters and R2, stored in https://tnc.box.com/s/828o0fmtezpo3v3s3u2ljpq03yd8qyn6

2.2 Collection of subset of important variables<br /> 
Within the subset_of_variables() function, the hyperparameters with the highest R2 are extracted and used to train the RF model. On each run (we used 40), we selected a subset of important variables using the variable importance score for each month.  The results are stored in a variable_importance.csv file. 

2.3 Final model training and testing<br /> 
We used the best performing hyperparameters and only the most important variables in the final model training.  The resulting model is saved and then tested with new data  from Nov 2021 - Apr 2022 that was not used in the training. We calculated the R-squared from the final model for each region - Xeric, Coastal Mountains and Interior Mountains respectively. 

**3. Operationally**<br /> 
The above steps are already coded in the script, operationally, you would need to pull the input training data [1.1.a] (you can update the box link whenever there are new training data) and run the Random Forest Model by inputting the number of iteration, number of runs, the month to run, example as follow:

<pre /> 
n_iter = 10<br /> 
number_of_runs = 40<br /> 
month_list = [1,2,3,4,5,6,7,8,9,10,11,12]<br /> 

for month in month_list:<br /> 
  print("Running hyperparameter search for month {}".format(month))<br /> 
  hyper_parameter_search(n_iter, month)<br /> 
  print("Selecting variable subset for month {}".format(month))<br /> 
  subset_of_variables(month, number_of_runs)<br /> 
  print("Training final model for month {}".format(month))<br /> 
  train_final_model(month)<br /> 



