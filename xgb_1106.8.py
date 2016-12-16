import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import KFold
from scipy.stats import skew, boxcox
from sklearn.preprocessing import StandardScaler
import itertools


shift = 200
COMB_FEATURE = 'cat80,cat87,cat57,cat12,cat79,cat10,cat7,cat89,cat2,cat72,cat81,cat11,cat1,cat13,cat9,cat3,cat16,cat90,cat23,cat36,cat73,cat103,cat40,cat28,cat111,cat6,cat76,cat50,cat5,cat4,cat14,cat38,cat24,cat82,cat25'.split(',')

def encode(charcode):
    r = 0
    ln = len(charcode)
    for i in range(ln):
        r += (ord(charcode[i])-ord('A')+1)*26**(ln-i-1)
    return r


fair_constant = 0.7
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess


def xg_eval_mae(yhat, dtrain):
    y = dtrain.get_label()
    return 'maeexp', mean_absolute_error(np.exp(y)-shift,
                                      np.exp(yhat)-shift)


def mungeskewed(train, test, numeric_feats):
    ntrain = train.shape[0]
    test['loss'] = 0
    train_test = pd.concat((train, test)).reset_index(drop=True)
    # compute skew and do Box-Cox transformation (Tilli)
    skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
    print("\nSkew in numeric features:")
    print(skewed_feats)
    skewed_feats = skewed_feats[skewed_feats > 0.25]
    skewed_feats = skewed_feats.index

    for feats in skewed_feats:
        train_test[feats] = train_test[feats] + 1
        train_test[feats], lam = boxcox(train_test[feats])
    return train_test, ntrain


if __name__ == "__main__":
    # print('Started')
    directory = ''
    # train = pd.read_csv(directory + 'train.csv')
    # test = pd.read_csv(directory + 'test.csv')
    # numeric_feats = [x for x in train.columns[1:-1] if 'cont' in x]
    # cats = [x for x in train.columns[1:-1] if 'cat' in x]
    # train_test, ntrain = mungeskewed(train, test, numeric_feats)
    
    # for comb in itertools.combinations(COMB_FEATURE, 2):
    #     feat = comb[0] + "_" + comb[1]
    #     train_test[feat] = train_test[comb[0]] + train_test[comb[1]]
    #     train_test[feat] = train_test[feat].apply(encode)
    #     print(feat)
    
    # cats = [x for x in train.columns[1:-1] if 'cat' in x]
    # for col in cats:
    #     train_test[col] = train_test[col].apply(encode)
    # train_test.loss = np.log(train_test.loss + shift)
    # ss = StandardScaler()
    # train_test[numeric_feats] = \
    #     ss.fit_transform(train_test[numeric_feats].values)
    # train = train_test.iloc[:ntrain, :].copy()
    # test = train_test.iloc[ntrain:, :].copy()
    # test.drop('loss', inplace=True, axis=1)
    # train.drop('loss', inplace=True, axis=1)

    # test.to_csv('test_combed.csv')
    # train.to_csv('train_combed.csv')

    train = pd.read_csv(directory+'train_combed.csv')
    test = pd.read_csv(directory+'test_combed.csv')
    raw_train = pd.read_csv(directory+'train.csv')
    loss = np.log(raw_train['loss']+200)

    print('read finished, start')

    xgb_params = {
        'seed': 2016,
        'colsample_bytree': 0.7,
        'silent': 1,
        'subsample': 0.7,
        'learning_rate': 0.03,
        'objective': 'reg:linear',
        'max_depth': 12,
        'min_child_weight': 100,
        'booster': 'gbtree',
    }

    best_nrounds = 100000  # 640 score from above commented out code (Faron)
    allpredictions = pd.DataFrame()
    train_preds = np.zeros(train.shape[0])
    train_preds_index = []
    kfolds = 10 # 10 folds is better!

    kf = KFold(train.shape[0], n_folds=kfolds,shuffle=True,random_state=123)
    for i, (train_index, test_index) in enumerate(kf):
        dtest = xgb.DMatrix(test[test.columns[2:]])
        print('Fold {0}'.format(i + 1))
        X_train, X_val = train.iloc[train_index], train.iloc[test_index]
        Y_train, Y_val = loss.iloc[train_index],loss.iloc[test_index] 

        dtrain = \
            xgb.DMatrix(X_train[X_train.columns[2:]],
                        label=Y_train)
        dvalid = \
            xgb.DMatrix(X_val[X_val.columns[2:]],
                        label=Y_val)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

        gbdt = xgb.train(xgb_params, dtrain, best_nrounds, watchlist,
                         obj=logregobj,
                         feval=xg_eval_mae, maximize=False,
                         verbose_eval=10,
                         early_stopping_rounds=100)

        train_preds[test_index] = np.exp(gbdt.predict(xgb.DMatrix(X_val[X_val.columns[2:]])))-200
        train_preds_index += test_index.tolist()    

        del dtrain
        del dvalid
        gc.collect()
        allpredictions['p'+str(i)] = \
            gbdt.predict(dtest, ntree_limit=gbdt.best_ntree_limit)
        del dtest
        del gbdt
        gc.collect()

    print(allpredictions.head())


    df_trian_preds = pd.DataFrame({'id':train_preds_index,'loss':train_preds})
    df_trian_preds.to_csv('xgb_train_preds.csv',index=False)

    submission = pd.read_csv(directory + 'sample_submission.csv')
    submission.iloc[:, 1] = \
        np.exp(allpredictions.mean(axis=1).values)-shift
    submission.to_csv('xgbmeansubmission.csv', index=None)
    submission.iloc[:, 1] = \
        np.exp(allpredictions.median(axis=1).values)-shift
    submission.to_csv('xgbmediansubmission.csv', index=None)

    print('Finished')