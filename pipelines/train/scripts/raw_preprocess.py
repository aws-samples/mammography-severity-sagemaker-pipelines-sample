
import argparse
import os
import glob
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Since we get a headerless CSV file, we specify the column names here.
feature_columns_names = [
    'BIRADS',
    'Age',
    'Shape',
    'Margin',
    'Density',
]

label_column = 'Severity'

feature_columns_dtype = {
    'BIRADS': np.float64,
    'Age': np.float64,
    'Shape': np.float64,
    'Margin': np.float64,
    'Density': np.float64,
}

#benign=0 or malignant=1 
label_column_dtype = {'Severity': bool}


def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing" 
    data_files = glob.glob(f"{base_dir}/input/*.csv")
    df = pd.concat((pd.read_csv(f) for f in data_files))
    df.replace("?", "NaN", inplace = True)
    df.columns = feature_columns_names + [label_column]
    feature_dtypes = merge_two_dicts(feature_columns_dtype, label_column_dtype)
    df = df.astype(feature_dtypes)
    
    numeric_features = list(feature_columns_names)
    numeric_transformer = Pipeline( 
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # This estimator allows different columns or column subsets of the input to be transformed separately and the features generated by each 
    # transformer will be concatenated to form a single feature space. This is useful for heterogeneous or columnar data, to combine several 
    # feature extraction mechanisms or transformations into a single transformer.
    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ]
    )

    y = df.pop("Severity") 

    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    pd.DataFrame(X_pre).to_csv(f"{base_dir}/baseline/baseline_dataset.csv", header=False, index=False)
