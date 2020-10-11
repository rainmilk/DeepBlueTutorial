from keras.layers import Input, Embedding, Dense, concatenate, Lambda
import keras.backend as K
from keras.models import Model


class NeuRegression:

    def __init__(self,
                 nb_feature,
                 nb_mv_type,
                 nb_age_grp,
                 nb_occupation):

        # Input layer
        a_in = Input(shape=[nb_feature], name='a_in')
        b_in = Input(shape=[1], name='b_in')
        c_in = Input(shape=[1], name='c_in')
        d_in = Input(shape=[1], name='d_in')

        # Layer 2
        feature_a = Dense(units=100, activation='relu')(a_in)
        embedding_b = Embedding(input_dim=nb_mv_type, output_dim=100, name='embedding_b')(b_in)
        embedding_c = Embedding(input_dim=nb_age_grp, output_dim=100, name='embedding_c')(c_in)
        embedding_d = Embedding(input_dim=nb_occupation, output_dim=100, name='embedding_d')(d_in)

        # Layer 3
        SqueezeEmbed = Lambda(lambda x: K.squeeze(x, 1))
        concat_h = concatenate([feature_a, SqueezeEmbed(embedding_b), SqueezeEmbed(embedding_c), SqueezeEmbed(embedding_d)])

        # Layer 4
        h_6 = Dense(units=200, activation='tanh',  name='h_6')(concat_h)

        # Output Layer
        y = Dense(units=1, activation='sigmoid',  name='output')(h_6)

        self.model = Model(inputs=[a_in, b_in, c_in, d_in], outputs=[y], name='neural_regression')









