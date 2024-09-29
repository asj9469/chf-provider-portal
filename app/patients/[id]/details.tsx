"use client";

import React from 'react'
import { useRouter } from "next/navigation";
import LineGraph from './LineGraph';
import 'chartjs-adapter-moment';

import { Patient, PatientProps } from '@/components/interfaces';
type DataPoint = { date: Date; value: number };

function populateArrays(
  patients: Patient[],
  weightData: { date: Date, value: number }[],
  systolicData: { date: Date, value: number }[],
  diastolicData: { date: Date, value: number }[],
  heartRateData: { date: Date, value: number }[],
  walkingDistanceData: { date: Date, value: number }[],
  fluidData: { date: Date, value: number }[]
) {
  patients.forEach(patient => {
    // const date = new Date(patient['Date'])
    const date = new Date(patient.date);

    weightData.push({
      date,
      value: Number(patient.weight_change),
    });

    systolicData.push({
      date,
      value: patient.systolicBP,
    });

    diastolicData.push({
      date,
      value: patient.diastolicBP,
    });

    heartRateData.push({
      date,
      value: patient.heartRate,
    });

    walkingDistanceData.push({
      date,
      value: patient.walkingDistance,
    });

    fluidData.push({
      date,
      value: patient.fluidIntake,
    });
  });
}

export default function PatientDetails({ patientData }: PatientProps) {
  const router = useRouter();

    const formatSeverity = (severity: number) => `${severity.toFixed(2)}%`;

    const formatExplanation = (explanation: string) => {
      // Replace **text** with <strong>text</strong> and \n with <br />
      return explanation
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Replace **text** with <strong>text</strong>
        .replace(/-\s*/g, '') // Remove hyphen (-) and any spaces after it
        .replace(/\n/g, '<br />'); // Replace \n with <br />
    };

    const serializedPatients: Patient[] = patientData.map((patient: any) => ({
      patientId: patient['Patient ID'],
      name: patient['Name'],
      age: patient['Age'],
      weight_change: patient['Weight Change'],
      systolicBP: patient['Systolic Blood Pressure (mmHg)'],
      diastolicBP: patient['Diastolic Blood Pressure (mmHg)'],
      heartRate: patient['Average Resting Heart Rate (bpm)'],
      walkingDistance: patient['Walking Distance (Steps)'],
      fluidIntake: patient['Fluid Intake Liters (Liters per Day)'],
      severity: patient['Severity'],
      explanation: patient['Explanation'],
      date: patient['Date'] ? patient['Date'].toString() : null, // Convert Date to string
    }));

    const sortedPatients = serializedPatients.sort((a, b) => {
      const dateA = new Date(a.date!); // Use the non-null assertion operator (!) since date is a string
      const dateB = new Date(b.date!);
      
      return dateA.getTime() - dateB.getTime(); // Ascending order: older dates first
    });

    // console.log(formatExplanation(serializedPatients[0].explanation))
    // console.log(serializedPatients[0].age)

    const weightData: DataPoint[] = [];
    const systolicData: DataPoint[] = [];
    const diastolicData: DataPoint[] = [];
    const heartRateData: DataPoint[] = [];
    const walkingDistanceData: DataPoint[] = [];
    const fluidData: DataPoint[] = [];

    populateArrays(serializedPatients, weightData, systolicData, diastolicData, heartRateData, walkingDistanceData, fluidData);

    return (
      <>
        <div className="min-w-full py-4">
        <span onClick={() => router.push('/patients')}
          className="text-blue-600 cursor-pointer hover:text-blue-800 px-10">
          &lt; Return to Patient List
        </span>

        <h1 className="text-2xl font-bold text-center mt-8">
          Detailed Report for {sortedPatients[sortedPatients.length - 1].name}
        </h1>

        <div className="mt-8 grid grid-cols-6 gap-6">
          <div className="col-start-2 col-end-6 inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
            <div className="rounded-lg border border-gray-100 py-3 shadow-sm lg:max-w-4xl mx-auto">
              <dl className="-my-3 divide-y divide-gray-100 text-sm">
                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Name</dt>
                  <dd className="text-gray-700 sm:col-span-2">{sortedPatients[sortedPatients.length - 1].name}</dd>
                </div>

                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Severity Level</dt>
                  <dd className="text-gray-700 sm:col-span-2">{formatSeverity(sortedPatients[sortedPatients.length - 1].severity)}</dd>
                </div>

                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Explanation</dt>
                  <dd
                    className="text-gray-700 sm:col-span-2"
                    dangerouslySetInnerHTML={{ __html: formatExplanation(sortedPatients[sortedPatients.length - 1].explanation) }}
                  />
                </div>
              
                <div className="mt-8">
                  <LineGraph data={weightData} label="Weight Change (lbs)" color="rgba(75, 192, 192, 1)" />
                  <LineGraph data={systolicData} label="Systolic BP (mmHg)" color="rgba(255, 206, 86, 1)" />
                  <LineGraph data={diastolicData} label="Diastolic BP (mmHg)" color="rgba(255, 99, 132, 1)" />
                  <LineGraph data={heartRateData} label="Heart Rate (bpm)" color="rgba(54, 162, 235, 1)" />
                  <LineGraph data={walkingDistanceData} label="Walking Distance (Steps)" color="rgba(153, 102, 255, 1)" />
                  <LineGraph data={fluidData} label="Fluid Intake (L)" color="rgba(255, 159, 64, 1)" />
                </div>
                </dl>
              </div>
          </div>
        </div>
      </div>
    </>
  );
}