// page.tsx

"use client";  // Add this line to indicate that this component is a Client Component

import { useState, ChangeEvent, FormEvent } from 'react';
import axios from 'axios';

interface Inputs {
  v1: string;
  v2: string;
  v3: string;
  v4: string;
  v5: string;
  v6: string;
  v7: string;
}

interface PredictionResponse {
  prediction: number;
}

const Home = () => {
  const [inputs, setInputs] = useState<Inputs>({
    v1: '',
    v2: '',
    v3: '',
    v4: '',
    v5: '',
    v6: '',
    v7: ''
  });
  const [prediction, setPrediction] = useState<number | null>(null);

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setInputs(prevState => ({ ...prevState, [name]: value }));
  };

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    const inputValues = Object.values(inputs).map(val => parseFloat(val)); // Convert to float
    try {
      const res = await axios.post<PredictionResponse>('http://localhost:5000/predict', { input: inputValues });
      setPrediction(res.data.prediction);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
  };

  return (
    <div>
      <h1>Prediction Model</h1>
      <form onSubmit={handleSubmit}>
        {Object.keys(inputs).map((key, index) => (
          <div key={index}>
            <label>{key}: </label>
            <input
              type="number"
              name={key}
              value={inputs[key as keyof Inputs]}
              onChange={handleInputChange}
            />
          </div>
        ))}
        <button type="submit">Get Prediction</button>
      </form>

      {prediction !== null && <h2>Prediction: {prediction}</h2>}
    </div>
  );
};

export default Home;

