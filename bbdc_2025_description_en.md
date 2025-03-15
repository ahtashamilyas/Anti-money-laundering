# Bremen Big Data Challenge 2025

## Data Download

The data is available for download: (File size approximately 134MB; unpacked approximately 373MB).

## Task Description

The Bremen Big Data Challenge (BBDC) 2025 focuses on the topic of Anti-money laundering. Specifically, the task involves predicting whether an individual has financial fraud, based on their transactional banking data for one month. For privacy and security reasons, the dataset is synthetically generated based on predefined distributions of fraudulent and non-fraudulent transacting users.

Participants in the competition receive transactional-level data for 11 thousand unique accounts, where approximately 15% users have committed fraud (specifically money-laundering). The task is to train a model which, based on the labelled training data, can predict which users of the unlabelled test set have committed fraud. For the student track, an additional aggregated dataset is provided as a starting point. This set contains data already at a user-level, with some useful aggregations performed on the original transactional data. Participants may use this set to build their foundational models, but are encouraged to create additional features from the transactional data, as there are some fraudulent behaviours which are not detectable based only on the aggregated set.

## Data Details

There are a total of 9 files, details of each of these are given below:

1. x_train.csv
2. x_train_aggregated.csv
3. y_test.csv
4. x_val.csv
5. x_val_aggregated.csv
6. y_val.csv
7. x_test.csv
8. x_test_aggregated.csv
9. student_skeleton.csv


1. "x_train.csv": Contains all the data necessary for the BBDC 2025 task. Each row depicts a single transaction. The details of the data are described below:
    
| Feature               | Description                                                                                                                                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AccountID             | Unique ID of the account holder. Predictions must be made at this level.                                                                                                                                                                                                              |
| Hour                  | Relative time reference of the transaction, in hours, since the start of the sample month. Every 24 hours represents a new day.                                                                                                                                                       |
| Action                | The type of transaction being performed. Descriptions of each transaction type are given below.                                                                                                                                                                                       |
| External              | Unique identifier of the other party in the transaction. The first letter describes the type of account. C-codes indicate the transaction interacted with another customer account, B-codes mean the interaction was with the bank itself, and M- codes depict a merchant or company. |
| Amount                | The numerical value of the transaction.                                                                                                                                                                                                                    |
| OldBalance            | The balance of the account holder before the transaction.                                                                                                                                                                                                                             |
| NewBalance            | The balance of the account holder after the transaction.                                                                                                                                                                                                                              |
| UnauthorizedOverdraft | If ‘1’, then the transaction was not processed because there were insufficient funds in the account to process the transaction.                                                                                                                                                     |

| Action   | Description                                                                                           |
| -------- | ----------------------------------------------------------------------------------------------------- |
| CASH_IN   | The account owner deposits cash into their account.                                                   |
| CASH_OUT  | The account owner withdraws cash from their account.                                                  |
| DEBIT    | Money is deducted from the account via an automated debit order. This action is performed through a bank (B).                                     |
| PAYMENT  | The account makes a digital payment through their physical or virtual card. This corresponds to a purchase made at a Merchant (M), who recieves the amount payed by the account owner.                          |
| TRANSFER | The account holder transfers money from their account to another’s (C), using their banking application. |

2. "x_train_aggregated.csv": An aggregated format of the transactional set. Contains some standard metrics, intended as a starting point for model development. Each row represents a single account.
   
| Feature         | Description                                                                    |
| --------------- | ------------------------------------------------------------------------------ |
| AccountID       | Unique ID of the account holder. Predictions must be made at this level.       |
| NumTransactions | The total number of transactions performed by this account in the given month. |
| AvgAmount       | The average value of a transaction for this account.                           |
| MaxAmount       | The maximum value of any type of transaction for this account.                 |
| TotalIn         | The total amount deposited in cash into the account over the month.                    |
| TotalOut        | The total amount spent by the account over the amount.                         |
| MaxIn           | The maximum amount deposited in cash into the account in a single transaction.          |
| MaxCashOut      | The largest cash withdrawal made by the account.                               |
| MaxDebit        | The largest debit order out of the account.                                    |
| MaxPayment      | The largest single card payment made by the account.                           |
| MaxTransfer     | The largest outgoing transfer made by the account.                             |
| MaxFreqH        | The largest number of transactions made by the account within any given hour.  |
| MaxFreqD        | The largest number of transactions made by the account within any given day.   |

3. "y_train.csv": This file shows which accounts from the training sample did or did not commit financial fraud. This binary flag is the target field for training the model:
   | Feature   | Description                                                                                             |
   | --------- | ------------------------------------------------------------------------------------------------------- |
   | AccountID | Unique subject ID.                                                                                      |
   | Fraudster | Binary classifier. 1 indicates that the account holder did commit fraud, 0 indicates that they did not. |

4-6. "Validation sets": For convenience, we have generated separated training and validation sets, as separating out a portion of the data can become complex when considering interaction networks between transactors and 3rd parties. For each of the training sets 1-3 above, there is an equivalent validation set. This should be used for validating and fine tuning trained models.

7-8. “Testing sets”: The testing sets provide the inputs to feed into the trained model(s) for submission. The provided fields and descriptions match those given in (1) and (2).

## Submission

The file “student_skeleton.csv" must be filled with the predicted fraud flags. This file can then be uploaded to the BBDC 2025 Submission Portal (https://bbdc.csl.uni-bremen.de/submission/). Subsequently, the score will be automatically calculated and displayed, and the leaderboard ranking will be updated. The number of rows and the order of the `AccountID` should not be changed.

## Scoring

The final score is calculated based on the f1-score. The higher the score, the better. The minimum score is 0.0 (0%), and the maximum score is 1.0 (100%).
