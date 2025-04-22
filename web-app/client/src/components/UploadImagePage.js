import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, ArrowLeft, ImageIcon } from 'lucide-react';
import axios from 'axios';

const UploadImagePage = () => {
  // ---------------- BCC State ----------------
  const [bccDragActive, setBccDragActive] = useState(false);
  const [bccFileSelected, setBccFileSelected] = useState(false);
  const [isUploadingBcc, setIsUploadingBcc] = useState(false);

  // ---------------- SCC State ----------------
  const [sccDragActive, setSccDragActive] = useState(false);
  const [sccFileSelected, setSccFileSelected] = useState(false);
  const [isUploadingScc, setIsUploadingScc] = useState(false);

  const navigate = useNavigate();

  // ---------------- Common Handlers ----------------
  const handleDrag = (event, setDragActive) => {
    event.preventDefault();
    event.stopPropagation();
    if (event.type === 'dragenter' || event.type === 'dragover') {
      setDragActive(true);
    } else {
      setDragActive(false);
    }
  };

  // ---------------- BCC Handlers ----------------
  const handleBccDrop = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    setBccDragActive(false);

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setBccFileSelected(true);
      await uploadBccFile(file);
    }
  };

  const handleBccFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setBccFileSelected(true);
      await uploadBccFile(file);
    }
  };

  const uploadBccFile = async (file) => {
    setIsUploadingBcc(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('is_bcc', 'true'); // <-- added this
  
      const response = await axios.post('http://localhost:5001/upload', formData);
  
      navigate('/select-image', {
        state: {
          islands: response.data.islands,
          is_bcc: true // or false
        }
      });
    } catch (error) {
      console.error('Error uploading BCC file:', error);
      setBccFileSelected(false);
    } finally {
      setIsUploadingBcc(false);
    }
  };  

  // ---------------- SCC Handlers ----------------
  const handleSccDrop = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    setSccDragActive(false);

    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSccFileSelected(true);
      await uploadSccFile(file);
    }
  };

  const handleSccFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSccFileSelected(true);
      await uploadSccFile(file);
    }
  };

  const uploadSccFile = async (file) => {
    setIsUploadingScc(true);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('is_bcc', 'false'); // <-- added this
  
      const response = await axios.post('http://localhost:5001/upload', formData);
  
      navigate('/select-image', {
        state: {
          islands: response.data.islands,
          is_bcc: false // or false
        }
      });
    } catch (error) {
      console.error('Error uploading SCC file:', error);
      setSccFileSelected(false);
    } finally {
      setIsUploadingScc(false);
    }
  };
  

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated background (same as your original) */}
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

      {/* Main content */}
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
          <div className="text-gray-300 font-medium">GatorVision</div>
        </nav>

        <div className="flex-1 flex flex-col items-center justify-center px-4 py-24">
          <div className="max-w-5xl w-full">
            <h2 className="text-5xl sm:text-6xl lg:text-7xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 animate-gradient-text bg-clip-text text-transparent text-center mb-12 py-2">
              Upload Pathology Slides
            </h2>

            {/* ---------- BCC Upload ---------- */}
            <div className="mb-16">
              <h3 className="text-3xl font-semibold text-center mb-6">
                BCC (Basal Cell Carcinoma)
              </h3>
              <div
                className={`border-2 border-dashed rounded-3xl p-16 transition-all duration-300 relative
                  ${bccDragActive || bccFileSelected
                    ? 'border-blue-400 bg-blue-500/10'
                    : 'border-gray-700 hover:border-gray-600'}
                `}
                onDragEnter={(e) => handleDrag(e, setBccDragActive)}
                onDragLeave={(e) => handleDrag(e, setBccDragActive)}
                onDragOver={(e) => handleDrag(e, setBccDragActive)}
                onDrop={handleBccDrop}
              >
                <div className="flex flex-col items-center justify-center gap-8">
                  <div className="w-24 h-24 rounded-full bg-blue-500/10 flex items-center justify-center">
                    {bccFileSelected ? (
                      <ImageIcon className="w-12 h-12 text-blue-400 animate-pulse" />
                    ) : (
                      <Upload className="w-12 h-12 text-blue-400" />
                    )}
                  </div>

                  <div className="text-center">
                    <p className="text-2xl text-gray-300 mb-3">
                      {bccFileSelected ? 'Processing...' : 'Drag and drop your BCC slide here'}
                    </p>
                    <p className="text-lg text-gray-400">
                      {bccFileSelected
                        ? 'Please wait while we process your image.'
                        : 'or click to browse'}
                    </p>
                  </div>

                  <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleBccFileChange}
                    accept="image/*"
                    disabled={isUploadingBcc}
                  />
                </div>
              </div>
            </div>

            {/* ---------- SCC Upload ---------- */}
            <div className="mb-16">
              <h3 className="text-3xl font-semibold text-center mb-6">
                SCC (Squamous Cell Carcinoma)
              </h3>
              <div
                className={`border-2 border-dashed rounded-3xl p-16 transition-all duration-300 relative
                  ${sccDragActive || sccFileSelected
                    ? 'border-blue-400 bg-blue-500/10'
                    : 'border-gray-700 hover:border-gray-600'}
                `}
                onDragEnter={(e) => handleDrag(e, setSccDragActive)}
                onDragLeave={(e) => handleDrag(e, setSccDragActive)}
                onDragOver={(e) => handleDrag(e, setSccDragActive)}
                onDrop={handleSccDrop}
              >
                <div className="flex flex-col items-center justify-center gap-8">
                  <div className="w-24 h-24 rounded-full bg-blue-500/10 flex items-center justify-center">
                    {sccFileSelected ? (
                      <ImageIcon className="w-12 h-12 text-blue-400 animate-pulse" />
                    ) : (
                      <Upload className="w-12 h-12 text-blue-400" />
                    )}
                  </div>

                  <div className="text-center">
                    <p className="text-2xl text-gray-300 mb-3">
                      {sccFileSelected ? 'Processing...' : 'Drag and drop your SCC slide here'}
                    </p>
                    <p className="text-lg text-gray-400">
                      {sccFileSelected
                        ? 'Please wait while we process your image.'
                        : 'or click to browse'}
                    </p>
                  </div>

                  <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleSccFileChange}
                    accept="image/*"
                    disabled={isUploadingScc}
                  />
                </div>
              </div>
            </div>

            <div className="text-center space-y-3">
              <p className="text-lg text-gray-400">
                Supported formats: PNG/JPG single-image slides or multi-image slides with multiple depths/sections
              </p>
              <p className="text-lg text-gray-400">Maximum file size: 20MB</p>
            </div>
          </div>
        </div>
      </div>

      {/* Loading Overlays */}
      {isUploadingBcc && (
        <div className="fixed inset-0 z-20 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center">
          <div className="animate-spin w-16 h-16 border-4 border-blue-400 border-t-transparent rounded-full mb-8" />
          <p className="text-2xl text-gray-300">Uploading your BCC image...</p>
        </div>
      )}

      {isUploadingScc && (
        <div className="fixed inset-0 z-20 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center">
          <div className="animate-spin w-16 h-16 border-4 border-blue-400 border-t-transparent rounded-full mb-8" />
          <p className="text-2xl text-gray-300">Uploading your SCC image...</p>
        </div>
      )}
    </div>
  );
};

export default UploadImagePage;
