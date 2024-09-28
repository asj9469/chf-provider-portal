// app/api/typeform-webhook/route.js

import { NextResponse } from 'next/server';
import dbConnect from '../../../utils/dbConnect';
import ResponseModel from '../../../models/Response';

// Function to transform Typeform payload to a structured document
function transformPayload(data) {
  const { event_id, event_type, form_response } = data;
  const { submitted_at, landed_at, token, form_id, answers } = form_response;

  const transformedAnswers = {};

  answers.forEach((answer) => {
    const { type, number, text, field } = answer;
    const questionTitle = field.title;

    let value;
    switch (type) {
      case 'number':
        value = number;
        break;
      case 'text':
        value = text;
        break;
      // Add more cases if there are other types
      default:
        value = null;
    }

    transformedAnswers[questionTitle] = value;
  });

  const document = {
    event_id,
    event_type,
    form_id,
    token,
    landed_at: new Date(landed_at),
    submitted_at: new Date(submitted_at),
    answers: transformedAnswers,
    received_at: new Date(),
  };

  return document;
}


export async function POST(request) {
    try {
      const data = await request.json();
      console.log('Receiving POST request at /api/typeform-webhook:', data);
  
      // Validate data
      if (!data || !data.form_response) {
        console.error('Invalid data format: Missing form_response');
        return NextResponse.json({ error: 'Invalid data format' }, { status: 400 });
      }
  
      // Transform the payload to a structured document
      const document = transformPayload(data);
      console.log('Transformed Document:', document);
  
      // Connect to MongoDB
      await dbConnect();
  
      // Wrap the insertion in a try-catch block to handle errors
      try {
        // Insert the document into the 'responses' collection
        const result = await ResponseModel.create(document);
  
        console.log('Data inserted with ID:', result._id);
  
        // Respond to Typeform
        return NextResponse.json({ status: 'success', insertedId: result._id }, { status: 200 });
      } catch (dbError) {
        console.error('Database insertion error:', dbError);
        return NextResponse.json({ error: 'Database Insertion Error' }, { status: 500 });
      }
    } catch (error) {
      console.error('Error handling webhook:', error);
      return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
  }