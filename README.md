# Customer churn prediction
Predict whether a customer will change telco provider.

For this purpose, we are going to use a telecommunication service dataset from a Kaggle competition. Kaggle is a competition platform which includes datasets for various 
topics.
## Table of Contents
* [1. Introduction](#1-introduction)
* [2. Data](#2-data)
* [3. Preprocessing](#3-preprocessing)
* [4. Results](#4-feature-engineering)
* [References](#references)

## 1. Introduction
Customer churn is estimated as the period of customer’s
contract in a service[1]. Customers’ retention also leads to
sales improvement and reduces marketing cost by avoiding
campaigns in already loyal customers [2].For this reason,
churn of customers end up as a serious problem.

## 2. Data
The training data set used in this experiment contains 4250
samples and 20 columns that describe customer churn for
telecommunication services.

ATTRIBUTES OF THE DATASET  </br>
| **Col. No** | **Attribute Name** | **Type** | **Description of the Attribute** |
| :--- | :--- | :--- | :--- |
| 1 | state | string | 2-letter code of the US state of customer residence. |
| 2 | account_length | numerical | Number of months the customer has been with the current telco provider. |
| 3 | area_code | string | 3 digit area code. |
| 4 | international_plan | boolean | The customer has international plan. |
| 5 | voice_mail_plan | boolean | The customer has voice mail plan. |
| 6 | number_vmail_messages | numerical | Number of voice-mail messages. |
| 7 | total_day_minutes | numerical | Total minutes of day calls. |
| 8 | total_day_calls | numerical | Total number of day calls. |
| 9 | total_day_charge | numerical | Total charge of day calls. |
| 10 | total_eve_minutes | numerical | Total minutes of evening calls. |
| 11 | total_eve_calls | numerical | Total number of evening calls. |
| 12 | total_eve_charge | numerical | Total charge of evening calls. |
| 13 | total_night_minutes | numerical | Total minutes of night calls. |
| 14 | total_night_calls | numerical | Total number of night calls. |
| 15 | total_night_charge | numerical | Total charge of night calls. |
| 16 | total_intl_minutes | numerical | Total minutes of international calls. |
| 17 | total_intl_calls | numerical | Total number of international calls. |
| 18 | total_intl_charge | numerical | Total charge of international calls. |
| 19 | number_customer_service_calls | numerical | Number of calls to customer service. |
| 20 | churn | boolean | Customer churn - target variable. |

For this experiment, we had to investigate our training
dataset and conclude on what we are going to use for the
preprocessing step.

Firstly, we examined the distribution for every pair of
features in our training dataset. Most of our columns’ values are normal
distributed except total international calls and number of
customers service calls that are skewed on the left because
most of their values are equal to zero.

![Values distribution](https://github.com/Dimstella/customer-churn-prediction/blob/main/Graphs/valuesDistribution.png) </br>

Next, we evaluated the contribution of every feature to the
churn. For the international plan and voice mail plan we had to
encode them to binary where ’yes’ equals to 1 and ’no’ equals
to 0. Most customers that have international plan are churn customers.

![](https://github.com/Dimstella/customer-churn-prediction/blob/main/Graphs/voicemail_international.png) </br>

For total day minutes, total eve minutes, total night minutes,
total day calls, total eve calls, and total night calls, we found
max and min and the median value for every column and
created three classes Mid, High, and Low. It is
interesting to mention that only total day minutes have a
significant impact on customers’ churn in contrast to other
features which are not correlated at all with the churn.

![](https://github.com/Dimstella/customer-churn-prediction/blob/main/Graphs/valuesClasses.png) </br>

Finally, we counted churn and no churn customers in order
to evaluate the balance of classes in our dataset. It is obvious 
the imbalance between the classes.

![](https://github.com/Dimstella/customer-churn-prediction/blob/main/Graphs/classesImbalance.png) </br>

## 3. Preprocessing

For the preprocessing, we used several methods in order to
transform our dataset’s features in a preferred shape according
to the analysis we contacted in the previous section.

We first used the label encoder method from the sklearn
library. With this method, we encoded the string values of
international plan and voice mail plan from ’yes’ and ’no’
to binary ’1’ and ’0’, respectively. The same procedure we
followed for the churn column.

By exploring the dataset features, we observe that we have
separate columns for minutes, charge, and calls through the
daytime. So, we combined nine columns into three by add up
them. Our final columns were minutes, charges, and calls that
accumulate the necessary information for our experiment. In
addition, we divided the minutes’ column with 60 in order to
create a new column that contains the time in hours.

Sequentially, we transformed our string values, state, and
area code into dummy variables. Dummy variables are independent
variables that take the value of either 0 or 1 and the 
usually used for categorical attributes. With this process, we
created two separated dataframes that concatenated them to
our main dataset.

Subsequently, we chose to not drop high correlated features
because we did not have any change to our results. We only
drop features that already preprocessed. These features were
area code and state with string values and minutes which
replaced with hours column.

Lastly, we faced a considerable imbalance of the churn
column. To overcome this
problem, we used an oversampling method called Random
Oversampling.

## 4. Results

The best performance is represented by ExtraTrees classifier which has
the best performance in classifying both churn and no churn
customers.

NOTE: Additional plots are provided in Graphs folder. </br> </br> 
SUMMARAZATION RESULTS </br>
| **Classifiers** | **Accuracy(%)**| **Precision(%)** | **Recall(%)** | **F-measure(%)** | **Specificity(%)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Random forest** | 98 | 99 | 96 | 97 | 99 |
| **Adaboost** | 94 | 95 | 92 | 94 | 95 |
| **Gradient Boosting** | 98 | 99 | 95 | 97 | 99 |
| **ExtraTrees** | 98 | 98 | 98.0 | 98 | 98 |
| **Decision Tree** | 96 | 96 | 96 | 96 | 95 |
| **XGBoost** | 97 | 99 | 95 | 97 | 99 |

![](https://github.com/Dimstella/customer-churn-prediction/blob/main/Graphs/classifiersComparison.png) </br>
