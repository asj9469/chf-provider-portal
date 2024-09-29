// import { getPatients } from "../../lib/mongodb/patients";
import clientPromise from "@/lib/mongodb";
import { NextApiRequest, NextApiResponse } from 'next';

export default async (req: NextApiRequest, res: NextApiResponse) => {
  try {
      const client = await clientPromise;
      const db = client.db("hackgt");
      const patients = await db
          .collection("movies")
          .find({})
          .sort({ metacritic: -1 })
          .limit(10)
          .toArray();
      res.json(patients);
  } catch (e) {
      console.error(e);
  }
}