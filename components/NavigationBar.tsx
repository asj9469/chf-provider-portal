"use client"
import React from "react";
import { useRouter } from "next/navigation";

export default function NavigationBar(){
    const router = useRouter();
    const handleClick = () => {
    router.push(`/`);
    };
    return(
        <div className="flex justify-between items-center bg-gray-100 py-5 px-12 w-full">
            <h1 className="text-2xl font-bold text-center">
                Healthy Jackets
            </h1>

            <button onClick={() => handleClick()}
            className="rounded-md bg-rose-600 px-4 py-2 text-sm font-semibold
            text-white shadow-sm hover:bg-rose-500 focus-visible:outline focus-visible:outline-2 
            focus-visible:outline-offset-2 focus-visible:outline-rose-600"
            >
                Logout
            </button>
        </div>
    )
}