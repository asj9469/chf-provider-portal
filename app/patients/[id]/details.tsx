"use client";

import React from 'react'
import NavigationBar from "@/components/NavigationBar";
import { useRouter } from "next/navigation";
import LineGraph from './LineGraph';
import { format } from 'date-fns';
import 'chartjs-adapter-moment';

const date = new Date();
const formattedDate = format(date, 'yyyy-MM-dd');

import { Patient, PatientDetailsProps } from '@/components/interfaces';

export default function PatientDetails({ patientId, patientData }: PatientDetailsProps) {
  const router = useRouter();

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
    date: patient['Date'] ? patient['Date'].toString() : null,
  }));

  // Sample data for the line graph (you can replace this with actual patient data)
  const weightData = [
    { date: new Date('2024-08-01'), value: 75 }, // Starting weight
    { date: new Date('2024-09-01'), value: 73.5 }, // Weight change over time
    { date: new Date('2024-09-02'), value: 72.5 },
    { date: new Date('2024-09-03'), value: 72 },
  ];

  const systolicData = [
      { date: new Date('2024-08-01'), value: 130 }, // Starting Systolic BP
      { date: new Date('2024-09-01'), value: 135 },
      { date: new Date('2024-09-02'), value: 138 },
      { date: new Date('2024-09-03'), value: 140 },
  ];

  const diastolicData = [
      { date: new Date('2024-08-01'), value: 85 }, // Starting Diastolic BP
      { date: new Date('2024-09-01'), value: 90 },
      { date: new Date('2024-09-02'), value: 92 },
      { date: new Date('2024-09-03'), value: 95 },
  ];

  const heartRateData = [
      { date: new Date('2024-08-01'), value: 75 }, // Starting heart rate
      { date: new Date('2024-09-01'), value: 80 },
      { date: new Date('2024-09-02'), value: 78 },
      { date: new Date('2024-09-03'), value: 82 },
  ];

  const walkingDistanceData = [
      { date: new Date('2024-08-01'), value: 3000 }, // Starting walking distance (steps)
      { date: new Date('2024-09-01'), value: 3500 },
      { date: new Date('2024-09-02'), value: 4000 },
      { date: new Date('2024-09-03'), value: 2000 },
  ];

  const fluidData = [
      { date: new Date('2024-08-01'), value: 2 }, // Fluid intake (liters)
      { date: new Date('2024-09-01'), value: 2.5 },
      { date: new Date('2024-09-02'), value: 3 },
      { date: new Date('2024-09-03'), value: 2.8 },
  ];

  return (
    <>
      <div className="min-w-full py-4">
        <span onClick={() => router.push('/patients')}
          className="text-blue-600 cursor-pointer hover:text-blue-800 px-10">
          &lt; Return to Patient List
        </span>

        <h1 className="text-2xl font-bold text-center mt-8">
          Detailed Report for {serializedPatients[0].name}
        </h1>

        <div className="mt-8 grid grid-cols-6 gap-6">
          <div className="col-start-2 col-end-6 inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
            <div className="rounded-lg border border-gray-100 py-3 shadow-sm lg:max-w-4xl mx-auto">
              <dl className="-my-3 divide-y divide-gray-100 text-sm">
                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Name</dt>
                  <dd className="text-gray-700 sm:col-span-2">{serializedPatients[0].name}</dd>
                </div>

                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Severity Level</dt>
                  <dd className="text-gray-700 sm:col-span-2">{serializedPatients[0].severity}</dd>
                </div>

                <div className="grid grid-cols-1 gap-1 p-3 odd:bg-gray-100 sm:grid-cols-3 sm:gap-4">
                  <dt className="font-medium text-gray-900">Explanation</dt>
                  <dd className="text-gray-700 sm:col-span-2">{serializedPatients[0].explanation}</dd>
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