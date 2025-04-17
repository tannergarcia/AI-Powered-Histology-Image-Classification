import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft } from 'lucide-react';
import CircularProgress from './CircularProgress';
import ImageModal from './ImageModal';

const PredictionsPage = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const { result, image_url, is_bcc } = location.state || {};

  const [isHeatmapHovered, setIsHeatmapHovered] = useState(false);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState('');
  const [modalAltText, setModalAltText] = useState('');

  const baseUrl = 'http://localhost:5001';

  if (!result || !image_url) {
    navigate('/');
    return null;
  }

  const { red_zone_coords, image_width, image_height } = result;
  const displayedImageWidth = 320;
  const displayedImageHeight = 320;
  const confidencePercentage = parseFloat((result.prediction * 100).toFixed(2));

  const scaleCoordinates = (coords) => {
    const [x, y, width, height] = coords;
    const scaleX = displayedImageWidth / image_width;
    const scaleY = displayedImageHeight / image_height;
    return [x * scaleX, y * scaleY, width * scaleX, height * scaleY];
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <div className="fixed inset-0 bg-black">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900 via-purple-900 to-blue-900 opacity-30 animate-gradient-x" />
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black to-black opacity-80" />
        <div className="absolute top-0 left-0 w-96 h-96 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-2000" />
        <div className="absolute bottom-0 left-20 w-96 h-96 bg-cyan-500 rounded-full mix-blend-multiply filter blur-3xl opacity-20 animate-blob animation-delay-4000" />
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-blue-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse" />
        <div className="absolute top-3/4 right-1/4 w-64 h-64 bg-purple-400 rounded-full mix-blend-multiply filter blur-3xl opacity-10 animate-pulse animation-delay-2000" />
      </div>

      <div className="relative min-h-screen backdrop-blur-xl flex flex-col">
        <nav className="absolute top-0 left-0 right-0 flex items-center justify-between p-6">
          <button
            onClick={() => navigate('/')}
            className="text-gray-300 hover:text-white flex items-center gap-2 transition-colors focus:outline-none"
          >
            <ArrowLeft className="w-6 h-6" />
            <span className="font-medium">Back</span>
          </button>
          <div className="text-gray-300 font-medium">Gator Vision</div>
        </nav>

        <div className="flex-1 flex flex-col items-center justify-center px-4 py-24">
          <div className="max-w-5xl w-full text-center">
            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold text-white mb-12 drop-shadow-lg">
              Analysis Result
            </h1>

            {/* Prediction Result & Model Info */}
            <div className="mb-16">
              <div className="inline-block px-8 py-6 bg-gradient-to-r from-blue-500 to-purple-500 rounded-3xl shadow-xl transform transition-transform duration-500 hover:scale-105">
                <p className="text-3xl font-bold text-white">
                  {result.label === 'Present' ? 'Cancerous Cells Detected' : 'No Cancerous Cells Detected'}
                </p>
                <div className="mt-4 text-gray-200 text-sm">
                  Model used:{' '}
                  <span className="font-semibold">
                    {is_bcc ? 'BCC (Basal Cell Carcinoma)' : 'SCC (Squamous Cell Carcinoma)'}
                  </span>
                </div>
              </div>
            </div>

            {/* Images */}
            <div className="flex flex-col lg:flex-row items-center justify-center gap-12 mb-16">
              {/* Original */}
              <div>
                <p className="text-xl text-gray-400 mb-4">Original Image</p>
                <img
                  src={`${baseUrl}${image_url}`}
                  alt="Analyzed Sample"
                  className="w-80 h-80 object-cover rounded-3xl shadow-2xl transform transition-transform duration-500 hover:scale-105 cursor-pointer"
                  onClick={() => {
                    setModalImageUrl(`${baseUrl}${image_url}`);
                    setModalAltText('Analyzed Sample');
                    setIsModalOpen(true);
                  }}
                />
              </div>

              {/* Heatmap */}
              {result.heatmap_url && (
                <div>
                  <p className="text-xl text-gray-400 mb-4">Grad‑CAM Heatmap</p>
                  <div
                    className="relative w-80 h-80 cursor-pointer"
                    onMouseEnter={() => setIsHeatmapHovered(true)}
                    onMouseLeave={() => setIsHeatmapHovered(false)}
                    onClick={() => {
                      setModalImageUrl(`${baseUrl}${result.heatmap_url}`);
                      setModalAltText('Grad‑CAM Heatmap');
                      setIsModalOpen(true);
                    }}
                  >
                    <img
                      src={`${baseUrl}${result.heatmap_url}`}
                      alt="Grad‑CAM Heatmap"
                      className="w-full h-full object-cover rounded-3xl shadow-2xl transform transition-transform duration-500 hover:scale-105"
                    />
                    {!isHeatmapHovered &&
                      red_zone_coords?.map((coords, index) => {
                        const [x, y, width, height] = scaleCoordinates(coords);
                        return (
                          <div
                            key={index}
                            className="absolute flex items-center justify-center"
                            style={{
                              left: `${x}px`,
                              top: `${y}px`,
                              width: `${width}px`,
                              height: `${height}px`,
                            }}
                          >
                            <div className="w-full h-full border-2 border-red-500 rounded-full animate-ping" />
                            <div className="absolute w-full h-full border-2 border-red-500 rounded-full" />
                          </div>
                        );
                      })}
                  </div>
                </div>
              )}
            </div>

            {/* Confidence Score */}
            <div className="my-16 w-full h-px bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500" />
            <div className="mb-16">
              <h2 className="text-4xl font-semibold text-gray-300 mb-8">Confidence Score</h2>
              <div className="flex items-center justify-center flex-col">
                <CircularProgress percentage={confidencePercentage} />
                <div className="text-xl text-gray-300 mt-4 font-sans">
                  {result.label === 'Present'
                    ? 'Cancerous Cells Detected'
                    : 'No Cancerous Cells Detected'}
                </div>
              </div>
            </div>

            {/* Buttons */}
            <div className="my-16 w-full h-px bg-gradient-to-r from-blue-500 via-purple-500 to-blue-500" />
            <div className="flex flex-col sm:flex-row items-center justify-center gap-6">
              <button
                onClick={() => navigate('/')}
                className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-8 py-4 rounded-xl text-xl font-semibold transition-all duration-300 transform hover:scale-105"
              >
                Analyze Another Sample
              </button>
              <button
                onClick={() => window.print()}
                className="border border-gray-500 hover:border-gray-400 text-gray-300 hover:text-white px-8 py-4 rounded-xl text-xl font-semibold transition-all duration-300 transform hover:scale-105"
              >
                Download Report
              </button>
            </div>
          </div>
        </div>

        <footer className="py-6 text-center text-gray-500 text-sm">
          © {new Date().getFullYear()} Gator Vision
        </footer>
      </div>

      <ImageModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        imageUrl={modalImageUrl}
        altText={modalAltText}
      />
    </div>
  );
};

export default PredictionsPage;
