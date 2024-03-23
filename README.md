# medxoom-2020

Practicum project for the Georgia Institute of Technology M.S. in Analytics. Completed project in December 2020 working with partner Mary Munro. 


## Project Overview

Medxoom is an Atlanta-based startup that aims to save customers, such as employer health
plans, money based on a comparison of pricing schemes at hospitals in the United States. For
this project, they wanted us to predict either the relative cost of inpatient and outpatient care
(RPIO), or the relative price of outpatient care (RPO) for these hospitals. Modeling and analysis
was performed separately so that effective models for each variable could be built. Medxoom
recommended binning these continuous predictors into groups, which allows for more tangible
comparisons. After building models, an error analysis was performed to allow us to gain insight
into model performance. The final section has outcomes and recommendations for future work


## Description of dataset and cleaning processes:

### Rand_hcris_cy_hosp_a_2020_05_01.csv.zip
   
a. The above file will be referred to as the “RAND dataset” going forward.

b. The RAND dataset contains information about the financial performance,
accounting, and other hospital operations data for a large number of hospitals in
the United States. In total there are roughly 1596 hospitals per year, with an
associated 1247 columns of variables for each hospital. Since there are a very
high number of variables in the RAND dataset, a method had to be devised to
reduce this to a tractable number. Several things were apparent about the RAND
dataset:

c. Many columns had 50-90% of values that were missing. The decision was made
to omit these columns as they would not provide much information

d. Some of the columns, such as “Inpatient Costs,” “Outpatient Costs,” “Total
Costs,” “Total Assets,” and “Total Salaries,” are combinations of many other
variables. These variables can be included while omitting their constituent parts,
while still yielding effectively the same information.

e. Geographical or location identifiers were mostly omitted, however a few of these a few 
of them like zip code, hospital name, hospital street address, state, were kept - not necessarily 
for predictive purposes, but they could help make sense of
regional variations or other patterns.

Using the above guidelines, a subset of 45 of the 1247 variables in RAND was
identified that still had the majority of information that was available. This file is
included as CuratedData.csv. A full list of these 45 variables is also included in
the appendix.

### Detailed_Data.xlsx

The Hospitals tab of the excel document contains the “labels” that Medxoom wants us to
predict. The variables “Relative Prices of Inpatient and Outpatient Care” and “Relative
Prices of Outpatient Care” were chosen to focus on for our analysis. This data was
joined with the RAND dataset using the Medicare provider number as a primary key to
produce the dataset used for Greg’s analysis.


### GregsData.csv

This dataset contains the results of joining the Hospitals tab of Detailed Data and the
RAND dataset by the medicare provider number, along with a few other modifications:
There were several Y/N responses that were changed to 0-1 encodings. Also the column
names were edited to reduce their length, for ease of model building in R. Lastly, the
data were filtered so that only the relevant years of 2015, 2016 and 2017 were included.
There are 4754 observations and 66 predictors total. This file had been included in the
zip folder.


## Exploratory Analysis and Variable Selection

Starting with GregsData.csv, an attempt was made to find the best subset of predictors for the
Relative Price for Outpatient Services (RPO). The reason this variable was chosen instead of
Relative Price for Inpatient and Outpatient Services (RPIO) was there exists more labeled data
points to train on - 1577 for RPO as opposed to 798 for RPIO. The reason for this is that we
hoped to obtain better predictions for more hospitals overall.

There are 66 variables to work with in GregsData.csv. However, not all of these are useful in the
modeling process. Some automated feature selection methods were performed to help us
choose a subset of these variables. First, correlation analysis was used to see if any of the
numeric predictors were highly correlated with RPO. The results of this are displayed visually:

Correlation Plot of Continuous Variables

![corrplot](https://github.com/schr0841/medxoom-2020/blob/main/images/corrplot.png)

In general it is best to avoid features with a high correlation to RPO, as this may lead to
problems in the modelling process such as multicollinearity. The predictors that are highly
correlated with RPO are RPIO, Standardized Price per Inpatient Stay and Standardized Price
per Outpatient Service. The choice was made to eliminate these variables from consideration.
Next, automated feature selection available within the R Caret package was applied. With RPO
as the response and all other variables as predictors, the output of the feature importance rating
is as follows:

## Model Creation and Analysis 

For this purpose the metric, FlipError, was defined as the fraction of test points which were classified further than an adjacent bucket:

$$
\text { FlipError }=\sum_{i=1}^n\left\{\mid \text{ predict }_i-\text{ ref }_i \mid>1\right\}
$$

We will use this metric in evaluation of model pperformance.


