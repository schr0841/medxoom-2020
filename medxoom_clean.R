#Read in csv file
library(readr)

#Imputation
library(mice)
library(Hmisc)

#Variable selection
library(caret)
library(corrplot)

#Model creation
library(h2o)



# PART 1 - LOAD DATASET

#Load Data
new_data2 <- read_csv("GregsData.csv")

# path="C:\Users\schre\OneDrive\Desktop\FlashDrive\Practicum\PractData\gtmsa_practicum_datasets"
#new_data2 <- read_csv("C:/Users/schre/OneDrive/Desktop/FlashDrive/Practicum/PractData/gtmsa_practicum_datasets/new_data2.csv")

ncol(new_data2)

#Split into 3 bins, representing Low, Medium and High relative prices
new_data2$RPO <- as.factor(as.numeric(cut_number(new_data2$Relative_Price_for_Outpatient_Services, 3)))
new_data2$RPIO <- as.factor(as.numeric(cut_number(new_data2$Relative_Price_for_Inpatient_and_Outpatient_Services, 3)))

#Split into 5 bins, representing Low, Medium-Low, Medium, Medium-High, and High relative prices
new_data2$RPO2 <- as.factor(as.numeric(cut_number(new_data2$Relative_Price_for_Outpatient_Services, 5)))
new_data2$RPIO2 <- as.factor(as.numeric(cut_number(new_data2$Relative_Price_for_Inpatient_and_Outpatient_Services, 5)))


# PART 2 - FEATURE SELECTION

#select only numeric columns of new_data2

nums <-  unlist(lapply(new_data2, is.numeric)) 
new_data3 <- new_data2[,nums]

#Drop constant columns
new_data3["Calendar_year"]=NULL

#Drop columns with missing values
new_data3["Cash_flow_margin"]=NULL
new_data3["Hosp_Compare_5_Star_Rat"]=NULL
new_data3["Num_ Inpatient_Stays"]=NULL
new_data3["outpat_costs_clinic"]=NULL
new_data3["Type_of_hospital_9.0"]=NULL
new_data3["critical_access_hosp_Y"]=NULL #these were complained about by PCA
new_data3["teach_hosp_hcr_Y"]=NULL

#Complete cases - needed for correlation matrix
new_data3<-new_data3[complete.cases(new_data3), ]


############ Other methods for feature selection
# Following this link 
# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/#:~:text=The%20caret%20R%20package%20provides,most%20important%20features%20for%20you.&text=How%20to%20select%20features%20from,the%20Recursive%20Feature%20Elimination%20method

# 1: Removing highly correlated variables


set.seed(7)
# load the library

# calculate and display correlation matrix
correlationMatrix <- round(cor(new_data3),2)
#print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

colorlegend(heat.colors(100), 0:47)
corrplot(correlationMatrix, method="circle", type="lower")

##############################################
#Rank features by importance
##############################################


# prepare training scheme
control <- trainControl(method="repeatedcv", number=10, repeats=3)

#Select numeric columns for analysis
nums <- unlist(lapply(new_data2, is.numeric))  

#Subset based on these columns, remove NAs
VIdata <- new_data2[,nums]
VIdata <- VIdata[complete.cases(VIdata),]
# train the model
model <- train(Relative_Price_for_Outpatient_Services ~ ., data=VIdata,  method="glmnet", trControl=control)
# estimate variable importance
importance <- varImp(model, scale=FALSE)
# summarize importance
#print(importance)
# plot importance
plot(importance)


##############################################
# Features Chosen as a result of these processes:
##############################################

colz<-c("Administrative_costs","Beds","Capital_asset_balances_total"
        ,"costs_total","employees_FTEs","Medicaid_charges","Medicaid_cost","impact_cmi","mdcr_inpat_charges"
        ,"mdcr_outpat_charges","net_income","operating_expenses"
        ,"operating_revenues","outpat_costs","total_assets","total_expenses","tot_liab_genfund"
        ,"teach_hosp_hcr_Y","Hosp_Compare_5_Star_Rat","Cost.to.charge_ratio_subtotal","total_margin",
        "inpat_length_of_stay","minor_teaching","Cash_flow_margin",
        "RPO","RPIO","RPO2","RPIO2",
        "City","State","Hospital_Name")

names(new_data2) <- make.names(names(new_data2))
new_data4=new_data2[which(new_data2$Calendar_year==2017),]

df<-new_data4[,colz]



###################################################
# PART 3 - IMPUTATION
#Impute missing values and create 2 additional datasets

#Find index of numeric columns: c(1:5,8:13,18:48,51:66)
impute_median <- df
numcols <- as.numeric(which(sapply(impute_median, is.numeric)))
impute_median<- data.matrix(impute_median[,numcols])

#Perform the imputation - simple median
for(i in 1:24){impute_median[is.na(impute_median[,i]), i] <- median(impute_median[,i], na.rm = TRUE)}
impute_median <- as.data.frame(impute_median)

#Add back City, State and Hospital Name as well as the 4 predictors
median_df <- cbind(impute_median,new_data2[,c(6,14,49,63:66)][1:1577,])

###################################################
#Perform imputation with MICE using the CART method 
mice_data<-df

#The MICE imputation generates a warning because we have 3 character columns, but you can disregard this as 
#these columns are not a source of NAs
impute_cart <-  mice(mice_data,m=5,maxit=10,meth='cart',seed=500)
mice_df<-complete(impute_cart,1)












# Reference:
# https://bradleyboehmke.github.io/HOML/stacking.html

#####################################################################
# PART 4 - MODEL CREATION
#####################################################################


####################
# (a) - 3 buckets 
####################

#Function for use in error analysis
return_error_3 <- function(test){
  test_df <- as.data.frame(test)
  
  
  #Subset the "big" errors - labeling 3 as 1, or 1 as 3
  subset_df <- subset(test_df,test_df$predictedRPO==1 & test_df$RPO==3 |
                        test_df$predictedRPO==3 & test_df$RPO==1)
  
  #Error Rate for 3 buckets
  err_3 <- nrow(subset_df)/nrow(test_df)
}


h2o.init()

# Run the next command to shut down the h2o instance
h2o.shutdown()

#Options are: unmodified dataframe (df), median imputation (median_df), MICE imputation (mice_df)
data = h2o.splitFrame(as.h2o(df),ratios=c(.7), seed=122)
#data = h2o.splitFrame(as.h2o(median_df),ratios=c(.7), seed=122)
#data = h2o.splitFrame(as.h2o(mice_df),ratios=c(.7), seed=122)

train_h2o <- data[[1]]
test <- data[[2]]

X <- names(df)[1:24]
Y <- names(df)[25]

###########################################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)

#Calculate performance based on confusion matrix
best_glm_perf <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf

acc_glm_3 <-1- best_glm_perf[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO<-predict[,1]
error_glm_3 <- return_error_3(test)
test$predictedRPO<-NULL


###########################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf

acc_rf_3 <-1- best_rf_perf[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO<-predict[,1]
error_rf_3 <- return_error_3(test)
test$predictedRPO<-NULL




###########################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2

acc_rf2_3 <-1- best_rf_perf2[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO<-predict[,1]
error_rf2_3 <- return_error_3(test)
test$predictedRPO<-NULL



###########################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf <- h2o.confusionMatrix(best_rf, newdata = test)
best_gbm_perf

acc_gbm_3 <-1- best_gbm_perf[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO<-predict[,1]
error_gbm_3 <- return_error_3(test)
test$predictedRPO<-NULL

###Stacking attempt

# Train a stacked tree ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "ensemble_tree",
  base_models = list(best_glm, best_rf, best_rf2,best_gbm),
  metalearner_algorithm = "AUTO"
)

ensemble_tree_perf <- h2o.confusionMatrix(best_gbm, newdata = test)
ensemble_tree_perf

#Save relevant predictions
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO<-predict[,1]
error_ensemble_3 <- return_error_3(test)




#################################################################################
#Error Analysis
#Analyze the errors in the prediction
test_df <- as.data.frame(test)

#Subset the "big" errors - labeling 3 as 1, or 1 as 3
subset_df <- subset(test_df,test_df$predictedRPO==1 & test_df$RPO==3 |
                      test_df$predictedRPO==3 & test_df$RPO==1)

#Error Rate for 3 buckets
err_3 <- nrow(subset_df)/nrow(test_df)

# Compare prediction errors to table of values for State
t1<-table(df$State)/sum(table(df$State))
df1<-as.data.frame(t1)
df_state<-data.frame(df1,row.names=1)
df_state<-t(df_state)

#Predictions
t2<-table(subset_df$State)/sum(table(subset_df$State))
df2<-as.data.frame(t2)
df_results<-data.frame(df2,row.names=1)
df_results<-t(df_results)

#This first row of this df is the percentage of data each state contributes to the data
#The second row of this df is the percentage of data each state contributes to the "large" misclassifications
df_combined_3buckets <- rbind(df_results,df_state[,colnames(df_results)])
rownames(df_combined_3buckets)<-c("Pct of datapoints in misclassification","Pct of datapoints responsible for")




####################
# (b) - 5 buckets 
####################

#Function for use in 5-bucket error analysis
return_error_5 <- function(test){
  #Error Analysis
  #Analyze the errors in the prediction
  test_df <- as.data.frame(test)
  
  #Subset the "big" errors:
  #1 as 5, 5 as 1
  #1 as 4, 4 as 1
  #2 as 5, 5 as 2
  #3 as 5, 5 as 3
  #2 as 4, 4 as 2
  #1 as 3, 3 as 1
  
  subset_df2 <- subset(test_df,test_df$predictedRPO2==1 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==1|
                         test_df$predictedRPO2==1 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==1|
                         test_df$predictedRPO2==2 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==2|
                         test_df$predictedRPO2==3 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==3|
                         test_df$predictedRPO2==2 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==2|
                         test_df$predictedRPO2==1 & test_df$RPO2==3 |test_df$predictedRPO2==3 & test_df$RPO2==1)
  
  
  #Error Rate for 5 buckets
  
  err_5 <- nrow(subset_df2)/nrow(test_df)
}



X <- names(df)[1:24]
Y <- "RPO2"


### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
###################################################################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)


best_glm_perf <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf


acc_glm_5 <-1- best_glm_perf[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO2<-predict[,1]
error_glm_5 <- return_error_5(test)
test$predictedRPO2<-NULL


###################################################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf

acc_rf_5 <-1- best_rf_perf[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf_5 <- return_error_5(test)
test$predictedRPO2<-NULL


###################################################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2

acc_rf2_5 <-1- best_rf_perf2[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf2_5 <- return_error_5(test)
test$predictedRPO2<-NULL


###################################################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf <- h2o.confusionMatrix(best_rf, newdata = test)
best_gbm_perf

acc_gbm_5 <-1- best_gbm_perf[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO2<-predict[,1]
error_gbm_5 <- return_error_5(test)
test$predictedRPO2<-NULL


###Stacking attempt
#DOES 

### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
# Train a stacked tree ensemble
ensemble_tree <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o, model_id = "ensemble_tree",
  base_models = list(best_glm, best_rf, best_rf2,best_gbm),
  metalearner_algorithm = "AUTO"
)

ensemble_tree_perf <- h2o.confusionMatrix(ensemble_tree, newdata = test)
ensemble_tree_perf


#Save relevant predictions
predict <- h2o.predict(object = ensemble_tree, newdata = test)
test$predictedRPO2<-predict[,1]
error_ensemble_5 <- return_error_5(test)



#Error Analysis
#Analyze the errors in the prediction
test_df <- as.data.frame(test)

#Subset the "big" errors:
#1 as 5, 5 as 1
#1 as 4, 4 as 1
#2 as 5, 5 as 2
#3 as 5, 5 as 3
#2 as 4, 4 as 2
#1 as 3, 3 as 1

subset_df2 <- subset(test_df,test_df$predictedRPO2==1 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==1|
                      test_df$predictedRPO2==1 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==1|
                      test_df$predictedRPO2==2 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==2|
                       test_df$predictedRPO2==3 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==3|
                       test_df$predictedRPO2==2 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==2|
                       test_df$predictedRPO2==1 & test_df$RPO2==3 |test_df$predictedRPO2==3 & test_df$RPO2==1)


#Error Rate for 5 buckets

err_5 <- nrow(subset_df2)/nrow(test_df)

#Predictions
t2<-table(subset_df2$State)/sum(table(subset_df2$State))
df2<-as.data.frame(t2)
df_results<-data.frame(df2,row.names=1)
df_results<-t(df_results)

#This first row of this df is the percentage of data each state contributes to the data
#The second row of this df is the percentage of data each state contributes to the "large" misclassifications
df_combined_5buckets <- rbind(df_results,df_state[,colnames(df_results)])
rownames(df_combined_5buckets)<-c("Pct of misclassifications","Pct of dataset")


#Visualization
plot(df_combined_5buckets[1,],df_combined_5buckets[2,],xlim=c(0,0.15),ylim=c(0,0.15),
     ylab="Representation in Data",xlab="Error in Prediction")
title("State Representation in Data vs State Error in Prediction")
abline(a=0,b=1,lty = 2)
abline(a=-0.025,b=1,lty = 2,col="red")
text(df_combined_5buckets[1,], df_combined_5buckets[2,], labels=colnames(df_combined_5buckets), cex=1,pos=4)








###############################################################################
# (c) Creating a separate model for states WA,LA,TN then combining
###############################################################################

###############################################################################
# (1) 3 buckets
###############################################################################


# 3 options: use non-imputed, median-imputed or mice-imputed data. 
# Default is non-imputed because we see the best performance with it

df_sub<-df[which(df$State=="WA"|df$State=="LA"|df$State=="TN"),]
df_rest<-df[which(df$State!="WA"& df$State!="LA" & df$State!="TN"),]

#df_sub<-median_df[which(median_df$State=="WA"|median_df$State=="LA"|median_df$State=="TN"),]
#df_rest<-median_df[which(median_df$State!="WA"& median_df$State!="LA" & median_df$State!="TN"),]

#df_sub<-mice_df[which(mice_df$State=="WA"|mice_df$State=="LA"|mice_df$State=="TN"),]
#df_rest<-mice_df[which(mice_df$State!="WA"& mice_df$State!="LA" & mice_df$State!="TN"),]



# WA, LA, and TN:
data = h2o.splitFrame(as.h2o(df_sub),ratios=c(.7), seed=124)
train_h2o <- data[[1]]
test <- data[[2]]

X <- names(df)[1:24]
Y <- names(df)[25]


#################
###############
###############


###########################################################
# Train & cross-validate a GLM model
### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)


best_glm_perf_sub3 <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf_sub3

acc_glm_3_sub3 <-1- best_glm_perf_sub3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO<-predict[,1]
error_glm_3_sub3 <- return_error_3(test)
test$predictedRPO<-NULL


###########################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf_sub3 <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf_sub3

acc_rf_3_sub3 <-1- best_rf_perf_sub3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO<-predict[,1]
error_rf_3_sub3 <- return_error_3(test)
test$predictedRPO<-NULL




###########################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2_sub3 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2_sub3

acc_rf2_3_sub3 <-1- best_rf_perf2_sub3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO<-predict[,1]
error_rf2_3_sub3 <- return_error_3(test)
test$predictedRPO<-NULL



###########################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf_sub3 <- h2o.confusionMatrix(best_rf, newdata = test)
best_gbm_perf_sub3

acc_gbm_3_sub3 <-1- best_gbm_perf_sub3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO<-predict[,1]
error_gbm_3_sub3 <- return_error_3(test)



###############
###############
###############



# Take data from the 'best' performing model above:

predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO2<-predict[,1]
test<-as.data.frame(test)
test1<-test
subset_df_1 <- subset(test,test$predictedRPO==1 & test$RPO==3|test$predictedRPO==3 & test$RPO==1)





#################################################################
#The "rest" of the data, excluding the 3 states from before

data = h2o.splitFrame(as.h2o(df_rest),ratios=c(.7), seed=124)
#data = h2o.splitFrame(as.h2o(median_df),ratios=c(.7), seed=124)
#data = h2o.splitFrame(as.h2o(mice_df),ratios=c(.7), seed=124)

train_h2o <- data[[1]]
test <- data[[2]]

X <- names(df)[1:24]
Y <- names(df)[25]


#################
###############
###############

### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
###########################################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)


best_glm_perf_rest3 <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf_rest3

acc_glm_3_rest3 <-1- best_glm_perf_rest3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO<-predict[,1]
error_glm_3_rest3 <- return_error_3(test)
test$predictedRPO<-NULL


###########################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf_rest3 <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf_rest3

acc_rf_3_rest3 <-1- best_rf_perf_rest3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO<-predict[,1]
error_rf_3_rest3 <- return_error_3(test)
test$predictedRPO<-NULL




###########################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2_rest3 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2_rest3

acc_rf2_3_rest3 <-1- best_rf_perf2_rest3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO<-predict[,1]
error_rf2_3_rest3 <- return_error_3(test)
test$predictedRPO<-NULL



###########################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf_rest3 <- h2o.confusionMatrix(best_rf, newdata = test)
best_gbm_perf_rest3

acc_gbm_3_rest3 <-1- best_gbm_perf_rest3[4,4]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO<-predict[,1]
error_gbm_3_rest3 <- return_error_3(test)



###############
###############
###############



predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
test<-as.data.frame(test)
test2<-test
subset_df_2 <- subset(test,test$predictedRPO==1 & test$RPO==3|test$predictedRPO==3 & test$RPO==1)

subset_df_3bins <- rbind(subset_df_1,subset_df_2)


test_comb <- rbind(test1,test2)

err_3_separate <- nrow(subset_df_3bins)/nrow(test_comb)








###############################################################################
# (2) 5 buckets
###############################################################################



data = h2o.splitFrame(as.h2o(df_sub),ratios=c(.7), seed=124)
#data = h2o.splitFrame(as.h2o(median_df),ratios=c(.7), seed=124)
#data = h2o.splitFrame(as.h2o(mice_df),ratios=c(.7), seed=124)

train_h2o <- data[[1]]
test <- data[[2]]
n1<-nrow(test)


X <- names(df)[1:24]
Y <- "RPO2"


#################
###############
###############

### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
###########################################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)


best_glm_perf_sub5 <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf_sub5

acc_glm_5_sub5 <-1- best_glm_perf_sub5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO2<-predict[,1]
error_glm_5_sub5 <- return_error_5(test)
test$predictedRPO2<-NULL


###########################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf_sub5 <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf_sub5

acc_rf_5_sub5 <-1- best_rf_perf_sub5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf_5_sub5 <- return_error_5(test)
test$predictedRPO2<-NULL




###########################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2_sub5 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2_sub5

acc_rf2_5_sub5 <-1- best_rf_perf2_sub5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf2_5_sub5 <- return_error_5(test)
test$predictedRPO2<-NULL



###########################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf_sub5 <- h2o.confusionMatrix(best_gbm, newdata = test)
best_gbm_perf_sub5

acc_gbm_5_sub5 <-1- best_gbm_perf_sub5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO2<-predict[,1]
error_gbm_5_sub5 <- return_error_5(test)
test$predictedRPO2<-NULL


###############
###############
###############



predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
test_df<-as.data.frame(test)

table(test_df$RPO2,test_df$predictedRPO2)

subset_df_1 <- subset(test_df,test_df$predictedRPO2==1 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==1|
                       test_df$predictedRPO2==1 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==1|
                       test_df$predictedRPO2==2 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==2|
                       test_df$predictedRPO2==3 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==3|
                       test_df$predictedRPO2==2 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==2|
                       test_df$predictedRPO2==1 & test_df$RPO2==3 |test_df$predictedRPO2==3 & test_df$RPO2==1)





#######################################################################################
#The data with the 3 states removed
data = h2o.splitFrame(as.h2o(df_rest),ratios=c(.7), seed=124)


train_h2o <- data[[1]]
test <- data[[2]]
n2<-nrow(test)

X <- names(df)[1:24]
Y <- "RPO2"



#################
###############
###############

### THE FOLLOWING MODEL DOES NOT WORK - 2024.3.22. SKIPPING FOR NOW
###########################################################
# Train & cross-validate a GLM model
best_glm <- h2o.glm(
  x = X, y = Y, training_frame = train_h2o, alpha = 0.1,
  remove_collinear_columns = TRUE, nfolds = 10, fold_assignment = "Modulo",
  keep_cross_validation_predictions = TRUE, seed = 123
)


best_glm_perf_rest5 <- h2o.confusionMatrix(best_glm, newdata = test)
best_glm_perf_rest5

acc_glm_5_rest5 <-1- best_glm_perf_rest5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_glm, newdata = test)
test$predictedRPO2<-predict[,1]
error_glm_5_rest5 <- return_error_5(test)
test$predictedRPO2<-NULL


###########################################################
# Train & cross-validate a RF model
best_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, #mtries = 20,
  max_depth = 30, min_rows = 1, sample_rate = 0.4, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_rf_perf_rest5 <- h2o.confusionMatrix(best_rf, newdata = test)
best_rf_perf_rest5

acc_rf_5_rest5 <-1- best_rf_perf_rest5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf_5_rest5 <- return_error_5(test)
test$predictedRPO2<-NULL




###########################################################
# Train & cross-validate a second RF model with different parameters
best_rf2 <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o, ntrees = 200, mtries = 5,
  max_depth = 40, min_rows = 1, sample_rate = 0.8, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 30, stopping_metric = "RMSE",
  stopping_tolerance = 0
)

best_rf_perf2_rest5 <- h2o.confusionMatrix(best_rf2, newdata = test)
best_rf_perf2_rest5

acc_rf2_5_rest5 <-1- best_rf_perf2_rest5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_rf2, newdata = test)
test$predictedRPO2<-predict[,1]
error_rf2_5_rest5 <- return_error_5(test)
test$predictedRPO2<-NULL



###########################################################
# Train & cross-validate a GBM model
best_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o, ntrees = 100, learn_rate = 0.01,
  max_depth = 5, min_rows = 5, sample_rate = 0.5, nfolds = 10,
  fold_assignment = "Modulo", keep_cross_validation_predictions = TRUE,
  seed = 123, stopping_rounds = 100, stopping_metric = "misclassification",
  stopping_tolerance = 0
)

best_gbm_perf_rest5 <- h2o.confusionMatrix(best_rf, newdata = test)
best_gbm_perf_rest5

acc_gbm_5_rest5 <-1- best_gbm_perf_rest5[6,6]


#Save relevant predictions and find error
predict <- h2o.predict(object = best_gbm, newdata = test)
test$predictedRPO2<-predict[,1]
error_gbm_5_rest5 <- return_error_5(test)
test$predictedRPO2<-NULL


###############
###############
###############


predict <- h2o.predict(object = best_rf, newdata = test)
test$predictedRPO2<-predict[,1]
test_df<-as.data.frame(test)
subset_df_2 <- subset(test_df,test_df$predictedRPO2==1 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==1|
                        test_df$predictedRPO2==1 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==1|
                        test_df$predictedRPO2==2 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==2|
                        test_df$predictedRPO2==3 & test_df$RPO2==5 |test_df$predictedRPO2==5 & test_df$RPO2==3|
                        test_df$predictedRPO2==2 & test_df$RPO2==4 |test_df$predictedRPO2==4 & test_df$RPO2==2|
                        test_df$predictedRPO2==1 & test_df$RPO2==3 |test_df$predictedRPO2==3 & test_df$RPO2==1)


subset_df_5bins <- rbind(subset_df_1,subset_df_2)

err_5_separate <- nrow(subset_df_5bins)/(n1+n2)


