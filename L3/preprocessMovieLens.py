import pandas as pd
import numpy as np
import random


def create_rating_train_test_set(ds_path, prop=0.8, negprop=10):
    rat_df = pd.read_csv(ds_path, delimiter='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    rndidx = np.random.permutation(rat_df.shape[0])
    split = int(rat_df.shape[0] * prop)
    train_idx = rndidx[:split]
    test_idx = rndidx[split:]
    train_data = rat_df.iloc[train_idx]
    test_data = rat_df.iloc[test_idx]

    train_data.to_csv(path_or_buf="Train_Rating_%.2f.csv" % prop, index=False)
    test_data.to_csv(path_or_buf="Test__Rating_%.2f.csv" % prop, index=False)


def create_train_test_set(ds_path, prop=0.8, negprop=10):
    rat_df = pd.read_csv(ds_path, delimiter='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    rndidx = np.random.permutation(rat_df.shape[0])
    split = int(rat_df.shape[0] * prop)
    train_idx = rndidx[:split]
    test_idx = rndidx[split:]
    train_data = rat_df.iloc[train_idx]
    test_data = rat_df.iloc[test_idx]
    test_data.loc[:, 'rating'] = 1
    test_users = test_data.user_id.unique()
    docs = list(rat_df.item_id.unique())
    neg_list = [test_data]
    for u in test_users:
        udata = test_data[test_data.user_id == u]
        if len(udata) > 0:
            samples = random.sample(docs, min(len(docs), len(udata) * (negprop + 1)))
            samples = [s for s in samples if s not in udata.item_id]
            datalen = int(min(len(docs), len(udata) * negprop))
            neg_list.append(pd.DataFrame({'user_id': u, 'item_id': samples[:datalen], 'rating': -1, 'timestamp': 0}))
    test_data = pd.concat(neg_list)

    train_data.to_csv(path_or_buf="Train_CTR_%.2f.csv" % prop, index=False)
    test_data.to_csv(path_or_buf="Test_CTR_%.2f.csv" % prop, index=False)


def create_coldstart_set(ds_path, prop=0.2, negprop=10):
    rat_df = pd.read_csv(ds_path, delimiter='::', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
    item_set = list(rat_df.item_id.unique())
    cs_items = random.sample(item_set, int(len(item_set) * prop))
    test_idx = rat_df.item_id.isin(cs_items)
    test_data = rat_df[test_idx]
    train_data = rat_df[~test_idx]
    test_data.loc[:, 'rating'] = 1
    test_users = test_data.user_id.unique()
    neg_list = [test_data]
    docs = list(test_data.item_id.unique())
    for u in test_users:
        udata = test_data[test_data.user_id == u]
        if len(udata) > 0:
            samples = random.sample(docs, min(len(docs), len(udata)*(negprop+1)))
            samples = [s for s in samples if s not in udata.item_id]
            datalen = int(min(len(docs), len(udata) * negprop))
            neg_list.append(
                pd.DataFrame({'user_id': u, 'item_id': samples[:datalen], 'rating': -1, 'timestamp':0}))
    test_data = pd.concat(neg_list)
    train_data.to_csv(path_or_buf="Train_CTR_CS_%.2f.csv" % prop, index=False)
    test_data.to_csv(path_or_buf="Test_CTR_CS_%.2f.csv" % prop, index=False)


ds_path = 'ratings.dat'
create_rating_train_test_set(ds_path, 0.8)
create_train_test_set(ds_path, 0.9)
create_coldstart_set(ds_path, 0.1)