# Customer churn prediction
Predict whether a customer will change telco provider.

For this purpose, we are going to use a telecommunication service dataset from a Kaggle competition. Kaggle is a competition platform which includes datasets for various 
topics.
## Table of Contents
* [1. Introduction](#1-introduction)
* [2. Data](#2-data)
* [3. Methodology](#3-methodology)
  * [Preprocessing](#preprocessing)
  * [Models](#models)
  * [Metrics](#metrics)
* [4. Experiments and results](#4-feature-engineering)
  * [A. Random Forest](#random-forest)
  * [B. AdaBoost](#adaboost)
  * [C. Gradient Boost](#gradient-boost)
  * [D. Decision Tree](#decision-tree)
  * [E. Extra Trees](#extra-tree)
  * [F. XGBoost](#xgboost)
  * [G. Bagging](#bagging)
* [Conclusions](#conclusions)
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

