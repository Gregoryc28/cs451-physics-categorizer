'use client';
import { useState } from 'react';

export default function PhysicsTutorPrototype() {
  const [problem, setProblem] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleCategorize = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/categorize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: problem })
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error("API Error", error);
    }
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto p-8 font-sans">
      <h1 className="text-2xl font-bold mb-2">ENGenuity Lab: Tutor Dropzone</h1>
      <p className="text-gray-600 mb-6">Paste a student's physics problem below for instant AI categorization.</p>
      
      <textarea
        className="w-full p-4 border rounded-md mb-4 h-32"
        placeholder="e.g., A 5 kg block is dropped from a height of 10 meters..."
        value={problem}
        onChange={(e) => setProblem(e.target.value)}
      />
      
      <button 
        onClick={handleCategorize}
        disabled={loading}
        className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:opacity-50"
      >
        {loading ? 'Analyzing...' : 'Categorize Problem'}
      </button>

      {result && (
        <div className="mt-8 p-6 bg-gray-50 border rounded-md">
          <h2 className="text-xl font-semibold mb-2">
            Identified Topic: <span className="text-blue-600 font-bold">{result.topic}</span>
          </h2>
        </div>
      )}
    </div>
  );
}