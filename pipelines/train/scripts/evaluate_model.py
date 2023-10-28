
import json
import pathlib
import tarfile

import joblib
import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extraction_filter = getattr(tarfile, 'data_filter',(lambda member, path: member))
        tar.extractall()

    model = joblib.load("xgboost-model")

    test_path = "/opt/ml/processing/test/test.csv"
    df = pd.read_csv(test_path, header=None)
    
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    print ("y_test is", y_test) 
    print("y_test length is", len(y_test)) 

    X_test = xgboost.DMatrix(df.values)

    predictions = model.predict(X_test)
    
    print("predictions before changing to 1 and 0", predictions) 
    for i in range(len(predictions)):
        if predictions[i] >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
    
    print("predictions after changing to 1 and 0", predictions) 
    print("predictions length is", len(predictions)) 
    
    prediction_diff=[]
    for j in range(len(predictions)):
        prediction_diff.append(y_test[j] - predictions[j])
        
    count = 0
    for k in range(len(prediction_diff)):
        if prediction_diff[k] == 0: 
            count = count + 1
            
    print("count is", count) 
        
    auc = roc_auc_score(y_test, predictions)
    
    report_dict= {
        "classification_metrics": {
            "auc": {"value": auc},
        },
    }
    
    print ("auc is", auc) 
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
