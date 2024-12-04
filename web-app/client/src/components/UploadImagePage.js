import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, ArrowLeft, ImageIcon } from 'lucide-react';
import axios from 'axios';

const UploadImagePage = () => {
  const [dragActive, setDragActive] = useState(false);
  const [fileSelected, setFileSelected] = useState(false);
  const [isUploading, setIsUploading] = useState(false); // Added to manage upload state
  const navigate = useNavigate();

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setFileSelected(true);
      await uploadFile(file);
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setFileSelected(true);
      await uploadFile(file);
    }
  };

  const uploadFile = async (file) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Send the image to the backend
      const response = await axios.post('http://localhost:5001/upload', formData);

      // Navigate to SelectImagePage with the list of islands
      navigate('/select-image', { state: { islands: response.data.islands } });
    } catch (error) {
      console.error('Error uploading file:', error);
      setFileSelected(false);
      // Handle error (e.g., show a message to the user)
    } finally {
      setIsUploading(false);
    }
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

        <div className="flex-1 flex items-center justify-center px-4 py-24">
          <div className="max-w-5xl w-full">
            <h2 className="text-5xl sm:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 animate-gradient-text bg-clip-text text-transparent text-center mb-12 py-2">
              Upload Pathology Slide
            </h2>

            <div className="relative">
              <div
                className={`
                  border-2 border-dashed rounded-3xl p-16 transition-all duration-300 relative
                  ${dragActive || fileSelected
                    ? 'border-blue-400 bg-blue-500/10'
                    : 'border-gray-700 hover:border-gray-600'}
                `}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                <div className="flex flex-col items-center justify-center gap-8">
                  <div className="w-24 h-24 rounded-full bg-blue-500/10 flex items-center justify-center">
                    {fileSelected ? (
                      <ImageIcon className="w-12 h-12 text-blue-400 animate-pulse" />
                    ) : (
                      <Upload className="w-12 h-12 text-blue-400" />
                    )}
                  </div>

                  <div className="text-center">
                    <p className="text-2xl text-gray-300 mb-3">
                      {fileSelected ? 'Processing...' : 'Drag and drop your slide image here'}
                    </p>
                    <p className="text-lg text-gray-400">
                      {fileSelected
                        ? 'Please wait while we process your image.'
                        : 'or click to browse'}
                    </p>
                  </div>

                  <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleFileChange}
                    accept="image/*"
                    disabled={isUploading} // Disable input during upload
                  />
                </div>
              </div>
            </div>

            <div className="mt-10 text-center space-y-3">
              <p className="text-lg text-gray-400">Supported formats: Single-image slide or Multi-image slide with multiple depths/sections (PNG)</p>
              <p className="text-lg text-gray-400">Maximum file size: 20MB</p>
            </div>
          </div>
        </div>
      </div>

      {/* Loading Overlay */}
      {isUploading && (
        <div className="fixed inset-0 z-20 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center">
          <div className="animate-spin w-16 h-16 border-4 border-blue-400 border-t-transparent rounded-full mb-8" />
          <p className="text-2xl text-gray-300">Uploading your image...</p>
        </div>
      )}
    </div>
  );
};

export default UploadImagePage;
