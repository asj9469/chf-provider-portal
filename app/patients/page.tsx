import NavigationBar from "@/components/NavigationBar";
import PatientsList from "./patientsList";
import connect from '@/lib/mongodb/index'
import { Patient } from "@/components/interfaces";

function getUniquePatientsById(patients: any[]) {
  const uniquePatientsMap = new Map();

  patients.forEach((patient:any) => {
      const patientId = patient['Patient ID'];
      // const patientDate = new Date(patient.date);
      const patientDate = new Date(patient['Date'])

      // If the patient ID is already in the map, check the date
      if (uniquePatientsMap.has(patientId)) {
          const existingPatient = uniquePatientsMap.get(patientId);
          const existingPatientDate = new Date(existingPatient['Date']);

          // Keep the patient with the most recent date
          if (patientDate >= existingPatientDate) {
              uniquePatientsMap.set(patientId, patient);
          } 
      }else {
        // If the patient ID is not in the map, add it (including single occurrences)
        uniquePatientsMap.set(patientId, patient);
    }
  });
  return Array.from(uniquePatientsMap.values());
}

export default async function Patients() {
    const client = await connect
    const cursor = await client.db("admin").collection("actual_patients").find();
    const patients = await cursor.toArray()

    // const data = JSON.parse(JSON.stringify(patients))
    // const uniquePatients = getUniquePatientsById(patients);

    const data = JSON.parse(JSON.stringify(patients));
    const uniquePatients = getUniquePatientsById(data);
    
  return (
    <>
        <NavigationBar/>
         
        {/* we pass in the patients list here. the PatientsList component handles the displaying with the passed in array
            we do it this way because this page.tsx serves as a "backend" where it handles the server side stuff
            we can't have server side stuff and client side stuff happening in the same page (that's just how Next.js works)
            if you look at the patientsList.tsx, you'll see that it's marked as a client side component on line 1
        */}
        <PatientsList patientData={uniquePatients}/>
    </>
  );
}