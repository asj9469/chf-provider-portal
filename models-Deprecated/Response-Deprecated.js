// models/Response.js

import mongoose from 'mongoose';

const responseSchema = new mongoose.Schema({
  event_id: { type: String, required: true },
  event_type: { type: String, required: true },
  form_id: { type: String, required: true },
  token: { type: String, required: true },
  landed_at: { type: Date, required: true },
  submitted_at: { type: Date, required: true },
  answers: {
    'Patient ID': { type: Number, required: true },
    Name: { type: String, required: true },
    Age: { type: Number, required: true },
    Weight: { type: Number, required: true },
    'Systolic Blood Pressure (mmHg)': { type: Number, required: true },
    'Diastolic Blood Pressure (mmHg)': { type: Number, required: true },
    'Average Resting Heart Rate (bpm)': { type: Number, required: true },
    'Walking Distance (Steps)': { type: Number, required: true },
    'Fluid Intake Liters (Liters per Day)': { type: Number, required: true },
  },
  received_at: { type: Date, default: Date.now },
});

// Prevent model overwrite upon initial compile
export default mongoose.models.Response || mongoose.model('Response', responseSchema);
