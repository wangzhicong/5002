
import numpy as np




def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    denominator = np.array(actual) + np.array(predicted)

    return 2 * np.mean(np.divide(dividend, denominator, out=np.zeros_like(dividend), where=(actual!=0), casting='unsafe'))



from keras import backend as K
from keras.layers import Lambda
def smape_keras(y_true, y_pred):
    numerator = K.abs(y_true - y_pred)
    denominator = y_true + y_pred
    # K.divide(numerator, denominator)
    division = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])
    return 2 * K.mean(division)
