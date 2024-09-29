import NavigationBar from "@/components/NavigationBar";
import PatientsList from "./patientsList";
import connect from '@/lib/mongodb/index'

export default async function Patients() {
    const client = await connect
    const cursor = await client.db("admin").collection("actual_patients").find();
    const patients = await cursor.toArray()

    const data = JSON.parse(JSON.stringify(patients))
    
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