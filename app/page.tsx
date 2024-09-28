"use client"
import { useRouter } from "next/router";

interface Patient {
  id: number;
  name: string;
  severity: string;
}

export default function Home() {

  const patients: Patient[] = [
    { id: 1, name: "John Doe", severity: "High" },
    { id: 2, name: "Jane Smith", severity: "Medium" },
    { id: 3, name: "Alice Johnson", severity: "Low" },
  ];

  const handleClick = (id: number) => {
    alert(`Patient ${id} details clicked!`);
    // You can implement navigation to details page here
    // Example: router.push(`/patients/${id}`);
  };

  return (
    <div className="min-w-full py-4">
      <h1 className="flex justify-center text-2xl font-bold  my-10">Patient List</h1>
      <div className="inline-block min-w-full py-2 align-middle sm:px-6 lg:px-8">
        {patients.map((patient) => (
          <div
            key={patient.id}
            className="cursor-pointer border rounded-md my-4 mx-48 p-6 flex justify-between items-center hover:bg-gray-100"
            onClick={() => handleClick(patient.id)}
          >
            <span className="font-medium">{patient.name}</span>
            <span className="text-sm text-gray-500">
              Severity Level: {patient.severity}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}