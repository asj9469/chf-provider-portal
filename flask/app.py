from flask import Flask, request, jsonify
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(7, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Sigmoid(),
            nn.Linear(30, 1),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits
model = NeuralNetwork().to(device)

model.load_state_dict(torch.load('model.pth'))

model.eval()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  
    inputs = torch.tensor([data['input']]) 
    
    with torch.no_grad():
        prediction = model(inputs).item() 

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

