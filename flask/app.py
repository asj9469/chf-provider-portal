from flask import Flask, request, jsonify
from openai import OpenAI
import pandas as pd
import torch
import torch.nn as nn
from pymongo import MongoClient

client = MongoClient("mongodb+srv://hackgt:hackgt@hackgtcluster.fltr4.mongodb.net/?retryWrites=true&w=majority&appName=HackGTCluster")
db = client['admin']
collection = db['actual_patients']

# Set device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Define the neural network model
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

# Load the model
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Maximum absolute values for normalization
max_abs_values = {
    "age": 100.0,
    "Weight_Change_kg": 3.5,
    "Systolic_BP_mmHg": 185.198862,
    "Diastolic_BP_mmHg": 124.869868,
    "Heart_Rate_bpm": 139.759569,
    "Walking_Distance_steps": 10000.0,
    "Fluid_Intake_liters": 3.906368,
}

app = Flask(__name__)

def normalize_input(data):
    """Normalizes the input data based on predefined max absolute values."""
    for column in data.columns:
        if column in max_abs_values:
            data[column] = data[column] / max_abs_values[column]
    return data

def explanation(json_info, score):
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are doctor who knows about symptoms for Conjestive Heart Failure."},
            {
                "role": "user",
                "content": f"""The patient information is the following, 
                Age: {json_info["age"]},
                Weight_Change_kg: {json_info["Weight_Change_kg"]},
                Systolic_BP_mmHg: {json_info["Systolic_BP_mmHg"]},
                Diastolic_BP_mmHg: {json_info["Diastolic_BP_mmHg"]},
                Heart_Rate_bpm: {json_info["Heart_Rate_bpm"]},
                Walking_Distance_steps: {json_info["Walking_Distance_steps"]},
                Fluid_Intake_liters: {json_info["Fluid_Intake_liters"]}

                The patient has a severity score of the following out of 100: {100*min(score[0], 100)}%

                Tell me consisly why the patient has the score they have based on the above information.
                """
            }
        ]
    )


    return str(completion.choices[0].message.content)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Check if the received data is valid
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format, expected a JSON object."}), 400

        # Convert input JSON to DataFrame
        df_new_input = pd.DataFrame([data])  # Wrap in a list to create a DataFrame

        # Normalize the new input
        df_normalized = normalize_input(df_new_input)

        # Convert normalized DataFrame to tensor
        input_tensor = torch.tensor(df_normalized.values, dtype=torch.float32).to(device)

        # Make predictions using the model
        with torch.no_grad():
            output = model(input_tensor)

        # Convert output tensor to list and return as JSON
        response_data = output.cpu().numpy().tolist()  # Move output to CPU before converting
        score = response_data[0]
        response = {
            "data": response_data,
            "explanation": explanation(data, score)  # Add the explanation key-value pair
        }
        collection.insert_one(response)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

