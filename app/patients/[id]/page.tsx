import React from 'react'
import NavigationBar from "@/components/NavigationBar";
import PatientDetails from './details';
import connect from '@/lib/mongodb/index'
import { Patient } from '@/components/interfaces';

export default async function PatientDisplay({ params }: { params: { id: string } }) {
    const patientId = (params.id); // example of how to use id passed in

    const client = await connect
    const cursor = await client.db("admin").collection("actual_patients").find();
    const patients = await cursor.toArray()
    const data = JSON.parse(JSON.stringify(patients))
    
    
// const patient = patients.find((p) => p.id === patientId);

    return (
      <>
        <NavigationBar/>
        <PatientDetails patientId={patientId} patientData={data}/>
      </>
    );
}