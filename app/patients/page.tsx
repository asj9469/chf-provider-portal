import NavigationBar from "@/components/NavigationBar";
import PatientsList from "./patientsList";

interface Patient {
    id: number;
    name: string;
    severity: string;
    explanation: string;
  }
  // dummy data
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

export default function Patients() {
    // Fetch the mongodb data here
    // we need to extract a list of patient name, id (for routing, it can be uuid), and severity

  return (
    <>
        <NavigationBar/>
         
        {/* we pass in the patients list here. the PatientsList component handles the displaying with the passed in array
            we do it this way because this page.tsx serves as a "backend" where it handles the server side stuff
            we can't have server side stuff and client side stuff happening in the same page (that's just how Next.js works)
            if you look at the patientsList.tsx, you'll see that it's marked as a client side component on line 1
        */}
        <PatientsList patients={patients}/>
    </>
  );
}