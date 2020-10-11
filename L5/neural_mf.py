from keras.layers import Embedding, Lambda, Dense, Input, concatenate, dot, multiply
from keras.models import Model
import keras.backend as K


class NeuMF:

    def __init__(self,
                 nb_mv,
                 nb_mv_feature,
                 nb_user,
                 nb_age_grp,
                 model_name):
        # Input layer
        mv_feat_in = Input(shape=[nb_mv_feature], name='mv_feat_in')
        mv_in = Input(shape=[1], name='mv_in')
        age_grp_in = Input(shape=[1], name='age_grp_in')
        user_in = Input(shape=[1], name='nb_user')

        # Layer 2
        feature_mv = Dense(units=100, activation='relu')(mv_feat_in)
        embedding_mv = Embedding(input_dim=nb_mv, output_dim=100, name='embedding_mv')(mv_in)
        embedding_age = Embedding(input_dim=nb_age_grp, output_dim=100, name='embedding_age')(age_grp_in)
        embedding_user = Embedding(input_dim=nb_user, output_dim=100, name='embedding_user')(user_in)

        # Layer 3
        SqueezeEmbed = Lambda(lambda x: K.squeeze(x, 1))
        h_mv = concatenate([feature_mv, SqueezeEmbed(embedding_mv)])
        h_usr = concatenate([SqueezeEmbed(embedding_age), SqueezeEmbed(embedding_user)])

        # Layer 4
        h_mv = Dense(units=100, activation='relu')(h_mv)
        h_usr = Dense(units=100, activation='relu')(h_usr)

        # template method
        self.model = self._create_model(inputs=[mv_feat_in, mv_in, age_grp_in, user_in], h_mv=h_mv, h_usr=h_usr, name=model_name)

    def _create_model(self, inputs, h_mv, h_usr, name):
        return None


class NeuMF_V1(NeuMF):

    def _create_model(self, inputs, h_mv, h_usr, name):
        y = dot([h_mv, h_usr], axes=-1)
        return Model(inputs=inputs, outputs=[y], name=name)

        
class NeuMF_V2(NeuMF):

    def _create_model(self, inputs, h_mv, h_usr, name):
        h_mv_usr = multiply([h_mv, h_usr])
        h_feat = Dense(units=100, activation='relu')(h_mv_usr)
        y = Dense(units=1, name='y')(h_feat)
        return Model(inputs=inputs, outputs=[y], name=name)
