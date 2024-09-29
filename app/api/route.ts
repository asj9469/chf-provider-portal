import connect from '@/lib/mongodb/index'

export async function GET() {
  const client = await connect
  const cursor = await client.db("admin").collection("actual_patients").find();
  const patients = await cursor.toArray()
  return Response.json(patients)
}