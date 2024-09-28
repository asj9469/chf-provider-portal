import React from 'react'
import NavigationBar from "@/components/NavigationBar";
import PatientDetails from './details';

// import clientPromise from '@/lib/mongodb';
// import { ObjectId } from 'mongodb';

interface Patient {
  id: number;
  name: string;
  severity: string;
  explanation: string;
}
// example data
const patients: Patient[] = [
  { id: 1, name: "John Doe", severity: "High" , 
    explanation: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ultrices ligula sit amet dignissim hendrerit. Pellentesque sagittis odio turpis, at vestibulum ipsum semper ac. Aenean lorem orci, molestie quis ex a, suscipit blandit lectus. Aliquam varius pulvinar velit, a porttitor tortor fermentum ac. Praesent pellentesque varius aliquam. Sed vulputate faucibus metus, non consequat erat faucibus sit amet. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed lacinia vitae lectus eget suscipit. Integer bibendum porta est et condimentum. Sed sollicitudin dui id eleifend placerat. Cras sit amet facilisis mi. Curabitur dictum ante vitae vestibulum eleifend. Nam suscipit laoreet molestie."},

  { id: 2, name: "Jane Smith", severity: "Medium",
    explanation: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ultrices ligula sit amet dignissim hendrerit. Pellentesque sagittis odio turpis, at vestibulum ipsum semper ac. Aenean lorem orci, molestie quis ex a, suscipit blandit lectus. Aliquam varius pulvinar velit, a porttitor tortor fermentum ac. Praesent pellentesque varius aliquam. Sed vulputate faucibus metus, non consequat erat faucibus sit amet. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed lacinia vitae lectus eget suscipit. Integer bibendum porta est et condimentum. Sed sollicitudin dui id eleifend placerat. Cras sit amet facilisis mi. Curabitur dictum ante vitae vestibulum eleifend. Nam suscipit laoreet molestie."
   },
  { id: 3, name: "Alice Johnson", severity: "Low",
    explanation: "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam ultrices ligula sit amet dignissim hendrerit. Pellentesque sagittis odio turpis, at vestibulum ipsum semper ac. Aenean lorem orci, molestie quis ex a, suscipit blandit lectus. Aliquam varius pulvinar velit, a porttitor tortor fermentum ac. Praesent pellentesque varius aliquam. Sed vulputate faucibus metus, non consequat erat faucibus sit amet. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Sed lacinia vitae lectus eget suscipit. Integer bibendum porta est et condimentum. Sed sollicitudin dui id eleifend placerat. Cras sit amet facilisis mi. Curabitur dictum ante vitae vestibulum eleifend. Nam suscipit laoreet molestie."
   },
];

export default async function PatientDisplay({ params }: { params: { id: string } }) {
    const patientId = parseInt(params.id); // example of how to use id passed in
    const patient = patients.find((p) => p.id === patientId);

    // Fetch the mongodb data here
    // using the provided id from parameter, we must be able to extract data
    // this includes all the patient information & explanation
    // ******************************************************************

    // something like this to call mongodb

    // const client = await clientPromise;
    // const db = client.db("your-database-name"); // Specify your MongoDB database name
    // const patient = await db.collection("patients").findOne({ _id: new ObjectId(params.id) });

    // verify if patient exists (just in case someone puts wrong url) - this is necessary just in case

    // if (!patient) {
    //   return (
    //       <>
    //           <NavigationBar />
    //           <div className="min-w-full py-4">
    //               <h1 className="flex justify-center text-2xl font-bold my-10">Patient Not Found</h1>
    //           </div>
    //       </>
    //   );
    // }

    return (
      <>
        <NavigationBar/>
        <PatientDetails patientId={patientId} patientData={patients}/>
      </>
    );
}