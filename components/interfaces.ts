export interface Patient {
    // id: string;
    patientId: number;
    name: string;
    age: number;
    weight_change: string;
    systolicBP: number;
    diastolicBP: number;
    heartRate: number;
    walkingDistance: number;
    fluidIntake: number;
    severity: number;
    explanation: string;
    date: string;
  }

  export interface PatientProps {
    patientData: Patient[]
  }
  export interface PatientDetailsProps {
    patientId: string;
    patientData: Patient[]; // patientData is typed as an array of Patient
  }