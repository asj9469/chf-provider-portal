import { NextApiRequest, NextApiResponse } from 'next';
import { MongoClient } from 'mongodb';

// Define the structure for the patient data
type PatientData = {
  patientId: string;
  name: string;
  age: number;
  weight: number;
  systolicBloodPressure: number;
  diastolicBloodPressure: number;
  restingHeartRate: number;
  walkingDistance: number;
  fluidIntakeLiters: number;
};

// MongoDB connection
const client = new MongoClient(process.env.MONGO_DB_URI || '', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

async function saveToMongoDB(data: any) {
  try {
    await client.connect();
    const db = client.db('your-database-name'); // Change this to your actual database name
    const collection = db.collection('patients'); // Collection name
    await collection.insertOne(data);
  } catch (error) {
    console.error('Error saving to MongoDB:', error);
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    const typeformData = req.body;

    try {
      // Extract data from Typeform submission
      const patientData: PatientData = {
        patientId: typeformData.form_response.answers[0]?.answer || '',
        name: typeformData.form_response.answers[1]?.answer || '',
        age: parseInt(typeformData.form_response.answers[2]?.answer, 10) || 0,
        weight: parseFloat(typeformData.form_response.answers[3]?.answer) || 0,
        systolicBloodPressure: parseInt(typeformData.form_response.answers[4]?.answer, 10) || 0,
        diastolicBloodPressure: parseInt(typeformData.form_response.answers[5]?.answer, 10) || 0,
        restingHeartRate: parseInt(typeformData.form_response.answers[6]?.answer, 10) || 0,
        walkingDistance: parseInt(typeformData.form_response.answers[7]?.answer, 10) || 0,
        fluidIntakeLiters: parseFloat(typeformData.form_response.answers[8]?.answer) || 0,
      };

      // Call the Flask API with the Typeform data
      const flaskResponse = await fetch('http://localhost:5000/your-flask-endpoint', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(patientData),
      });

      if (!flaskResponse.ok) {
        throw new Error('Error from Flask API');
      }

      // Get the processed data from the Flask API
      const processedData = await flaskResponse.json();

      // Save the processed data to MongoDB
      await saveToMongoDB(processedData);

      res.status(200).json({ message: 'Data processed and saved successfully' });
    } catch (error) {
      console.error('Error processing webhook:', error);
      res.status(500).json({ error: 'Failed to process Typeform submission' });
    }
  } else {
    res.status(405).json({ message: 'Only POST requests are allowed' });
  }
}
