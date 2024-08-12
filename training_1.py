import random
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
intense = json.loads(open('intense.json').read())
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intense['intense']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
# print(documents)

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
# print(words)
classes = sorted(set(classes))
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_pattern = [lemmatizer.lemmatize(word.lower()) for word in document[0]]
    for word in words:
        bag.append(1) if word in word_pattern else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_X = torch.tensor(training[:, 0].tolist(), dtype=torch.float32)
train_Y = torch.tensor(training[:, 1].tolist(), dtype=torch.float32)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(len(train_X[0]), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, len(train_Y[0]))
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6)

# Training loop
epochs = 400
batch_size = 8
num_batches = len(train_X) // batch_size

for epoch in range(epochs):
    model.train()
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_X = train_X[start:end]
        batch_Y = train_Y[start:end]

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y.argmax(dim=1))
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'chatbot_model.pth')
print("Model saved")
