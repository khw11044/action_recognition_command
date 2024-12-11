
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional

def get_model(actions, sequence_length, joints):

    model = Sequential()
    # 양방향 LSTM과 Dropout 추가
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu'), input_shape=(sequence_length, joints)))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
    model.add(Dropout(0.3))
    # Dense 레이어에 Dropout 추가
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    # 출력 레이어
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    return model