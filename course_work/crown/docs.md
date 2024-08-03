# Cross Validation Techniques

## Overview

Cross validation techniques are used to evaluate the performance of a model and ensure it generalizes well to unseen data. Two common methods are K-Fold Cross Validation and Leave-One-Out Validation (LOOV).

## 1. K-Fold Cross Validation

### Process

1. **Split the Dataset**: Divide the dataset into K equally sized folds.
2. **Training and Testing**: For each fold:
   - Train the model on K-1 folds.
   - Test the model on the remaining fold.
3. **Evaluate Performance**: Compute performance metrics for each fold.
4. **Average the Results**: Average the metrics across all K iterations.

### Data Loaders

- the folder `fold_dataloaders` has the pth files for all the folds, the content of the file is a dictionary having train_dataset and test_dataset as keys to access the respective dataloaders.

- Useage of the pth files.
```python
import os 
from dataloader_1 import *

for fold in os.listdir(r'F:\TANISHQ\ML-hands-on\course_work\crown\data\fold_data'):
    file=r'F:\TANISHQ\ML-hands-on\course_work\crown\data\fold_data'+'\\'+fold
    train,test=load_fold_data(file)
    ##trainin loop as usual 


    ##testing on the test data

    ##printing the results for the particualr fold(averaged over the entire fold)

##printing the results for the entire dataset(averaged over all the folds)
```


## 2. Leave-One-Out Validation (LOOV)

### Process

1. **Split the Dataset**: Each data point is treated as a single fold.
2. **Training and Testing**: For each data point:
   - Train the model on all other data points.
   - Test the model on the single data point.
3. **Evaluate Performance**: Compute performance metrics for each iteration.
4. **Average the Results**: Average the metrics across all iterations.

### Data Loaders

- **Training DataLoader**: Created using all data points except one.
- **Validation DataLoader**: Created using the single excluded data point.

By using these techniques, we ensure that our model's performance is evaluated in a comprehensive and unbiased manner.