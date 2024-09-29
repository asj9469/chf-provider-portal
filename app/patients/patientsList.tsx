"use client"
import { useRouter } from "next/navigation";
import { Patient, PatientProps } from "@/components/interfaces";

export default function PatientsList({patientData}: PatientProps) {
  
  const router = useRouter();
  const handleClick = (id: number) => {
    router.push(`/patients/${id}`);
  };

  const serializedPatients: Patient[] = patientData.map((patient: any) => ({
    patientId: patient['Patient ID'],
    name: patient['Name'],
    age: patient['Age'],
    weight_change: patient['Weight'],
    systolicBP: patient['Systolic Blood Pressure (mmHg)'],
    diastolicBP: patient['Diastolic Blood Pressure (mmHg)'],
    heartRate: patient['Average Resting Heart Rate (bpm)'],
    walkingDistance: patient['Walking Distance (Steps)'],
    fluidIntake: patient['Fluid Intake Liters (Liters per Day)'],
    severity: patient['Severity'],
    explanation: patient['Explanation'],
    date: patient['Date'] ? patient['Date'].toString() : null, // Convert Date to string
  }));

  return (
    <>
    <div className="min-w-full py-4">
      <h1 className="flex justify-center text-2xl font-bold my-10">Congestive Heart Failure Patient List</h1>

      <div className="mt-8 grid grid-cols-6 gap-6">
        <div className="col-start-2 col-end-6 inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
        <div className="w-full px-4 sm:px-6 lg:px-8">
            {/* {patientData.map((patient: any) => (
            <div
                key={patient.id}
                className="cursor-pointer border rounded-md my-4 p-6 lg:max-w-2xl mx-auto flex justify-between items-center bg-white hover:bg-gray-100"
                onClick={() => handleClick(patient.id)}
            >
                <span className="font-medium">{patient.name}</span>
                <span className="text-sm text-gray-500">
                Severity Level: {patient.severity}
                </span>
            </div>
            ))} */}
        </div>
        </div>
      </div>
    </div>
    </>
    
  );
}