import React from 'react'
import NavigationBar from "@/components/NavigationBar";
import PatientDetails from './details';
import connect from '@/lib/mongodb/index'

export default async function PatientDisplay({ params }: { params: { id: string } }) {
    const patientId = (params.id); // example of how to use id passed in

    const client = await connect
    const cursor = await client.db("admin").collection("actual_patients").find();
    const patients = await cursor.toArray()
    const data = JSON.parse(JSON.stringify(patients))
    
    const filteredPatients = data.filter((p: any) => p['Patient ID'] === parseInt(patientId));

    // If no patient is found, return a "Patient Not Found" message
    if (filteredPatients.length === 0) {
      return (
        <>
          <NavigationBar />
          <div className="min-w-full py-4">
            <h1 className="flex justify-center text-2xl font-bold my-10">Patient Not Found</h1>
          </div>
        </>
      );
    }

    return (
      <>
        <NavigationBar/>
        <PatientDetails patientData={filteredPatients}/>
      </>
    );
}