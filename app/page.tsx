"use client"
import { useRouter } from "next/navigation";
import { useState } from 'react';

export default function Home() {
  
  const router = useRouter();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const handleClick = () => {
    if (email === 'practitioner@hospital.com' && password === 'chf') {
      // Use router.push for redirection
      router.push('/patients');
    } else {
      alert('Incorrect email or password');
    }
    
  };

  return (
    <>
      <div className="flex justify-center text-5xl font-bold my-10 mt-40">Welcome to Healthy Jackets!</div>
      <div className="flex justify-center text-lg mt-4 leading-relaxed text-gray-500">Sign in with your provider account to access the Congestive Heart Failure patient list</div>
      
      <div className="mt-8 grid grid-cols-6 gap-6">
        <div className="col-start-3 col-end-5 rounded-lg bg-white p-10 shadow-sm">
        
          <div className="px-10">
            <label htmlFor="Email" className="block text-md font-medium text-gray-700"> Email </label>

            <input
              type="email"
              id="Email"
              name="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 p-3 w-full rounded-md border bg-white text-md text-gray-700 shadow-sm"
            />
          </div>
          <div className="px-10 pt-5 pb-10">
            <label htmlFor="Password" className="block text-md font-medium text-gray-700"> Password </label>

            <input
              type="password"
              id="Password"
              name="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 p-3 w-full rounded-md border bg-white text-md text-gray-700 shadow-sm"
            />
          </div>
          <div className="px-10">
          <button type="submit" onClick={() => handleClick()}
            className="flex w-full justify-center rounded-md bg-indigo-600 py-3 text-sm font-semibold leading-6
              text-white shadow-sm hover:bg-indigo-500 focus-visible:outline focus-visible:outline-2 
              focus-visible:outline-offset-2 focus-visible:outline-indigo-600">
            Sign in
          </button>
          </div>
        </div>
      </div>
    </>
  );
}