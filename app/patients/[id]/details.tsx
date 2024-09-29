"use client";

import React from 'react'
import NavigationBar from "@/components/NavigationBar";
import { useRouter } from "next/navigation";

import { Patient, PatientDetailsProps } from '@/components/interfaces';

export default function PatientDetails({patientId, patientData}: PatientDetailsProps) {

    const router = useRouter();
    // const patient = patientData.find((p:any) => p.id === patientId);
    
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

    console.log(serializedPatients[0].age)

    // if (!patient) {
    //   return (
    //     <>
    //         <div className="min-w-full py-4">
    //             <h1 className="flex justify-center text-2xl font-bold my-10">Patient Not Found</h1>
    //         </div>
    //     </>
    //   );
    // }
    

    return (
      <>
        <div className="min-w-full py-4">
        <span onClick={() => router.push('/patients')}
            className="text-blue-600 cursor-pointer hover:text-blue-800 px-10"
        >
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
                </dl>
              </div>
          </div>
        </div>
        </div>
      </>
    );
}