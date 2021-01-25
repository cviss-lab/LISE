import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(gamma=2, alpha=0.6):
    # https://github.com/Atomwh/FocalLoss_Keras
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:, 0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


def mape_0_to_1(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return K.mean(diff, axis=-1) * 1.

def weighted_mape(weight):
    def weighted_mape(y_true, y_pred):
        weights = y_true / weight
        diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                                K.epsilon(),
                                                None)) * weights
        return K.mean(diff, axis=-1) * 100.
    return weighted_mape
