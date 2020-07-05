from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def classifier_DNN_fn(input_shape, output_shape, hidden_shapes=[30]):
    '''
    クラス分類用DNN

    Parameters
    -----
    input_shape : int
    output_shape : int
    hidden_shapes : list of int
        default:[30]

    Returns
    -----
    model : tensorflow.keras.model
    '''
    model = Sequential()
    
    for i, hidden_shape in enumerate(hidden_shapes):
        if i == 0:
            model.add(Dense(hidden_shapes[0], input_shape=(input_shape, ), activation='relu'))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(hidden_shape, activation='relu'))
            model.add(Dropout(0.3))

    model.add(Dense(output_shape, activation='softmax'))
    return model
