from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, input_shape=(10,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()
