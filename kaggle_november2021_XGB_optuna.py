import numpy as np
import pandas as pd
import random
import os
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import optuna
import xgboost as xgb
import sklearn.metrics
from sklearn.model_selection import train_test_split


def objective(trial):
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x, label=test_y)

    param = {
        "silent": 1,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
        param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
        param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)

    # Add a callback for pruning.
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation-auc")
    bst = xgb.train(param, dtrain, evals=[(dtest, "validation")], callbacks=[pruning_callback])
    preds = bst.predict(dtest)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    return accuracy


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


SEED = 42
seed_everything(SEED)

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

X = data.drop('target', axis=1)
y = data['target']
del data

features = X.columns

continuous_features = []
discrete_features = []

for elem in features:
    if X[elem].dtype == 'float64':
        X[elem] = X[elem].astype('float32')
    else:
        X[elem] = X[elem].astype('uint8')

continuous_features = []
discrete_features = []

for elem in features[:-1]:
    if test_data[elem].dtype == 'float64':
        test_data[elem] = test_data[elem].astype('float32')
    else:
        test_data[elem] = test_data[elem].astype('uint8')

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 1e-2,
    'seed': SEED,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'n_estimators': 10000,
    'max_depth': 8,
    'alpha': 10,
    'lambda': 1e-1,
    'min_child_weight': 100,
}

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=70000, timeout=216000)
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

print(study.best_trial)
print(study.best_params)


# submission = pd.read_csv('sample_submission.csv')
# predictions = np.mean(np.column_stack(preds), axis=1)
# submission['target'] = predictions
# submission.to_csv('./submission_Nov21_xgb.csv', index=False)
