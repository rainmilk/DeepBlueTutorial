from L5.neural_regression import NeuRegression


neu_reg = NeuRegression(nb_feature=2, nb_mv_type=5, nb_age_grp=5, nb_occupation=10)
neu_reg.model.compile(optimizer='adam', loss='binary_crossentropy')
neu_reg.model.summary()
