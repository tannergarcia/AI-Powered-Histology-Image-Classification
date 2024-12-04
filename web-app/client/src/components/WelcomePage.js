// WelcomePage.js
import React from 'react';
import { ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const WelcomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 bg-black">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900 via-purple-900 to-blue-900 opacity-30 animate-gradient-x" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black to-black opacity-80" />
        
        {/* Animated orbs */}
        <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute bottom-0 left-20 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000" />
        
        {/* Additional floating elements */}
        <div className="absolute inset-0">
          <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse" />
          <div className="absolute top-3/4 right-1/4 w-64 h-64 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-2000" />
        </div>
      </div>

      {/* Content */}
      <div className="relative min-h-screen backdrop-blur-xl flex items-center justify-center">
        <div className="max-w-7xl mx-auto px-8 text-center">
        <h1 className="text-7xl sm:text-8xl lg:text-9xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 animate-gradient-text bg-clip-text text-transparent tracking-tight leading-tight mb-12">
            Gator Vision
          </h1>
          
          <p className="text-gray-400 text-1xl sm:text-2xl lg:text-3xl max-w-5xl mx-auto mb-16 leading-relaxed">
            Revolutionizing basal cell carcinoma detection through advanced artificial intelligence. 
            Experience precise, rapid, and reliable pathology slide analysis.
          </p>

          <button 
            className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-12 py-8 rounded-xl text-2xl font-semibold transition-all duration-300 hover:scale-105"
            onClick={() => navigate('/upload')}
          >
            Start Analysis
            <ArrowRight className="ml-3 h-7 w-7 inline" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default WelcomePage;