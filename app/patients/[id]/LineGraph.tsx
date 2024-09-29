// LineGraph.tsx
import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LinearScale, PointElement, LineElement, Tooltip, Legend, Title, TimeScale } from 'chart.js';
import 'chartjs-adapter-moment';

ChartJS.register(LinearScale, PointElement, LineElement, Tooltip, Legend, Title, TimeScale);

interface LineGraphProps {
  data: { date: Date; value: number }[];
  label: string;
  color: string;
}

const LineGraph: React.FC<LineGraphProps> = ({ data, label, color }) => {
  const chartData = {
    labels: data.map(point => point.date.toISOString().split('T')[0]), // Keep this for data mapping
    datasets: [
      {
        label,
        data: data.map(point => point.value),
        borderColor: color,
        backgroundColor: color,
        fill: false,
        tension: 0.1,
      },
    ],
  };

  const options = {
    scales: {
      x: {
        type: 'time',
        time: {
          unit: 'day', // Set to 'day' to display only the dates
          tooltipFormat: 'MMM D, YYYY', // Format for tooltips
          displayFormats: {
            day: 'MMM D', // Format to display on x-axis
          },
        },
        title: {
          display: true,
          text: 'Date',
        },
      },
      y: {
        title: {
          display: true,
          text: label,
        },
      },
    },
  };

  return <Line data={chartData} options={options} />;
};

export default LineGraph;