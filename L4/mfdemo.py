#%%
from keras.layers import Input, Embedding, dot, Lambda
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import pandas as pd


# 读入文件
df_train = pd.read_csv('ua.base', sep='\t')
df_test = pd.read_csv('ua.test', sep='\t')

user_ids = set(df_train.uid).union(set(df_test.uid))
item_ids = set(df_train.mid).union(set(df_test.mid))
nb_user = len(user_ids)
nb_item = len(item_ids)
print('Number of users = %d | Number of movies = %d' % (nb_user, nb_item))

u_id2idx = dict(zip(user_ids, range(nb_user)))
i_id2idx = dict(zip(item_ids, range(nb_item)))

# 替换ID为Index
df_train.assign(uid=[u_id2idx[uid] for uid in df_train.uid])
df_train.assign(mid=[i_id2idx[iid] for iid in df_train.mid])
df_test.assign(uid=[u_id2idx[uid] for uid in df_test.uid])
df_test.assign(mid=[i_id2idx[iid] for iid in df_test.mid])

#%%
# 构造MF模型
reg = l2(1e-5)
nb_latent_factor = 100

user_idx = Input(shape=[1], name='user_idx')
item_idx = Input(shape=[1], name='item_idx')

user_emb = Embedding(input_dim=nb_user, output_dim=nb_latent_factor, input_length=1, embeddings_regularizer=reg, name='user_embedding')
item_emb = Embedding(input_dim=nb_item, output_dim=nb_latent_factor, input_length=1, embeddings_regularizer=reg, name='item_embedding')

u_lf = user_emb(user_idx)
i_lf = item_emb(item_idx)

SqueezeEmbed = Lambda(lambda x: K.squeeze(x, 1))
u_lf = SqueezeEmbed(u_lf)
i_lf = SqueezeEmbed(i_lf)

pred_rating = dot([u_lf, i_lf], axes=-1)

mf = Model(inputs=[user_idx, item_idx], outputs=pred_rating, name='mf_model')
mf.compile(optimizer='adam', loss='mse', metrics=['mse'])
mf.summary()

#%%
# 模型训练
mf.fit(x=[df_train.uid.values, df_train.mid.values], y=df_train.rating.values, epochs=3, batch_size=64)

#%%
# 模型预测
p_ratings = mf.predict(x=[df_test.uid.values, df_test.mid.values])

# MAE
N = len(p_ratings)
MAE = np.sum(np.abs(p_ratings.flatten() - df_test.rating.values)) / N
print('MF Model MAE: %g' % MAE)
