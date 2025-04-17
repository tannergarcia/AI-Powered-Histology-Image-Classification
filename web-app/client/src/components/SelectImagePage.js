import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, CheckCircle } from 'lucide-react';
import axios from 'axios';

const SelectImagePage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { islands, is_bcc } = location.state || {};
  const baseUrl = 'http://localhost:5001';

  const [selectedImage, setSelectedImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [imageErrors, setImageErrors] = useState({});

  useEffect(() => {
    window.scrollTo(0, 0); // Scroll to top on component mount
  }, []);

  const handleImageSelection = (imageUrl) => {
    navigate('/progress', {
      state: {
        imageUrl,   // <- selected image
        is_bcc      // <- keep this flowing
      }
    });    
  };

  const handleImageError = (imageUrl) => {
    console.error(`Failed to load image: ${baseUrl}${imageUrl}`);
    setImageErrors((prev) => ({ ...prev, [imageUrl]: true }));
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated background (same as WelcomePage) */}
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
      <div className="relative min-h-screen backdrop-blur-xl flex flex-col">
        {/* Navigation Bar */}
        <nav className="absolute top-0 left-0 right-0 flex items-center justify-between p-6">
          <button
            onClick={() => navigate('/')}
            className="text-gray-300 hover:text-white flex items-center gap-2 transition-colors"
          >
            <ArrowLeft className="w-6 h-6" />
            <span className="font-medium">Back</span>
          </button>
          <div className="text-gray-300 font-medium">Gator Vision</div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 flex flex-col items-center justify-center px-4 py-24">
          <div className="max-w-7xl w-full text-center">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight mb-8">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-gradient-text">
                Select Your Sample
              </span>
            </h1>
            
            <p className="text-xl text-gray-400 mb-16">
              Your uploaded slide contains multiple samples. Please select one for analysis.
            </p>

            {islands.length === 0 ? (
              <p className="text-red-400 text-center mb-4">No images received</p>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-12 px-4">
                {islands.map((imageUrl, index) => (
  <div
    key={index}
    className={`
      relative group overflow-hidden cursor-pointer rounded-3xl
      transform transition-transform duration-500 hover:scale-105
      shadow-xl hover:shadow-2xl
      aspect-square
    `}
    onClick={() => handleImageSelection(imageUrl)}
  >
    {/* Image with error handling */}
    {!imageErrors[imageUrl] ? (
      <img
        src={`${baseUrl}${imageUrl}`}
        alt={`Sample ${index + 1}`}
        className="w-full h-full object-contain transition-transform duration-700 group-hover:scale-110"
        onError={() => handleImageError(imageUrl)}
      />
    ) : (
      <div className="w-full h-full flex items-center justify-center text-red-400">
        Failed to load image
      </div>
    )}

    {/* Overlay */}
    <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

    {/* Text Overlay */}
    <div className="absolute bottom-0 left-0 right-0 p-6 text-left text-white opacity-0 group-hover:opacity-100 transition-opacity duration-500">
      <h3 className="text-2xl font-semibold">Sample {index + 1}</h3>
      <p className="text-sm text-gray-200 mt-2">
        Click to select this sample for analysis.
      </p>
    </div>

    {/* Selection Indicator */}
    {selectedImage === imageUrl && (
      <div className="absolute inset-0 flex items-center justify-center bg-blue-500/30 backdrop-blur-sm">
        <CheckCircle className="w-20 h-20 text-blue-400 animate-bounce" />
      </div>
    )}
  </div>
))}

              </div>
            )}
          </div>
        </main>
      </div>

      {/* Loading Overlay */}
      {isLoading && (
        <div className="fixed inset-0 z-20 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center">
          <div className="animate-spin w-16 h-16 border-4 border-blue-400 border-t-transparent rounded-full mb-8" />
          <p className="text-2xl text-gray-300">Analyzing your sample...</p>
        </div>
      )}
    </div>
  );
};

export default SelectImagePage;
