from L5.neural_mf import NeuMF_V1, NeuMF_V2


neu_mf_v1 = NeuMF_V1(nb_mv=1000, nb_mv_feature=2, nb_user=2000, nb_age_grp=5, model_name='neuMF_V1')
neu_mf_v1.model.compile(optimizer='adam', loss='mse')
neu_mf_v1.model.summary()


neu_mf_v2 = NeuMF_V2(nb_mv=1000, nb_mv_feature=2, nb_user=2000, nb_age_grp=5, model_name='neuMF_V1')
neu_mf_v2.model.compile(optimizer='adam', loss='mse')
neu_mf_v2.model.summary()