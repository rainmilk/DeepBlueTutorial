import numpy as np
import pandas as pd
import scipy.sparse as sp

# 读入文件
df_train = pd.read_csv('ua.base', sep='\t')
df_test = pd.read_csv('ua.test', sep='\t')

n_users = df_train.uid.unique().shape[0]
n_items = df_train.mid.unique().shape[0]
print('Number of users = %d | Number of movies = %d' % (n_users, n_items))

# 使用三元组(ratings, (uid, mid))初始化一个稀疏矩阵
base_data_matrix = sp.coo_matrix((df_train.rating.values, (df_train.uid.values, df_train.mid.values))).toarray()
test_data_matrix = sp.coo_matrix((df_test.rating.values, (df_test.uid.values, df_test.mid.values))).toarray()

pred_mat = np.zeros_like(test_data_matrix)

# 这里需要自行设计，怎么更好计算相似度，思考一下这里可以改进的问题
eps = np.finfo(float).eps
for i in df_train.uid.unique():
    ln = base_data_matrix[i]
    tn = test_data_matrix[i]
    # Jaccard相似度，分母加上eps，避免除以0
    jaccard = np.sum(np.logical_and(ln, base_data_matrix), axis=1) / (np.sum(np.logical_or(ln, base_data_matrix), axis=1) + eps)
    nz = np.nonzero(tn)[0] # 找出要所有预测的items的id
    for j in nz:
        ref_ratings = base_data_matrix[:, j] # 找出所有对item j所有的ratings
        ref_users = np.nonzero(ref_ratings) # 找出对应的user
        jc_ref = jaccard[ref_users]  # 找出对应的similarities
        njc_ref = jc_ref / (np.sum(jc_ref) + eps)
        pred_rating = np.sum(njc_ref * ref_ratings[ref_users]) # 相似度 * 对应的user ratings
        pred_mat[i, j] = pred_rating
        
# 计算MAE
N = np.count_nonzero(test_data_matrix)
MAE = np.sum(np.abs(pred_mat - test_data_matrix)) / N
print('User-based CF MAE: %g' % MAE)