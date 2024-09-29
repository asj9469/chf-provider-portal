import NavigationBar from "@/components/NavigationBar";
import PatientsList from "./patientsList";
import connect from '@/lib/mongodb/index'
import { Patient } from "@/components/interfaces";

export default async function Patients() {
    const client = await connect
    const cursor = await client.db("admin").collection("actual_patients").find();
    const patients = await cursor.toArray()

    const serializedPatients: Patient[] = patients.map((patient: any) => ({
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

    const data = JSON.parse(JSON.stringify(serializedPatients))
    // console.log(data)
  return (
    <>
        <NavigationBar/>
         
        {/* we pass in the patients list here. the PatientsList component handles the displaying with the passed in array
            we do it this way because this page.tsx serves as a "backend" where it handles the server side stuff
            we can't have server side stuff and client side stuff happening in the same page (that's just how Next.js works)
            if you look at the patientsList.tsx, you'll see that it's marked as a client side component on line 1
        */}
        <PatientsList patients={data}/>
    </>
  );
}