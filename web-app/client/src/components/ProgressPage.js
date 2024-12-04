// ProgressPage.js
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import axios from 'axios';

const ProgressPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { imageUrl } = location.state || {};

  const baseUrl = 'http://localhost:5001';

  const [progress, setProgress] = useState(0);
  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
  const [apiResponseData, setApiResponseData] = useState(null);

  const apiCompleted = useRef(false);
  const apiCompletionTime = useRef(null);
  const startTime = useRef(Date.now());

  const messages = [
    'Starting analysis...',
    'Reading image data...',
    'Extracting tissue regions...',
    'Segmenting samples...',
    'Preprocessing images...',
    'Making predictions on input image...',
    'Interpreting model predictions...',
    'Loading trained model for Grad-CAM visualization...',
    'Extracting last convolutional layer outputs...',
    'Computing gradients for heatmap generation...',
    'Normalizing Grad-CAM heatmap...',
    'Generating overlay of heatmap on original image...',
    'Analyzing cellular structures...',
    'Detecting anomalies...',
    'Compiling results...',
    'Finalizing report...',
    'Analysis complete.',
  ];

  useEffect(() => {
    if (!imageUrl) {
      navigate('/select-image');
      return;
    }

    const updateProgress = () => {
      const now = Date.now();
      const elapsedTime = now - startTime.current;
      let newProgress = progress;

      if (!apiCompleted.current) {
        if (elapsedTime <= 3000) {
          // Progress from 0% to 50% over 3 seconds
          newProgress = (elapsedTime / 3000) * 50;
        } else {
          // After 3 seconds, increment progress towards 90%
          const timeSinceThreeSeconds = elapsedTime - 3000;
          const estimatedTotalTime = 11000; // Estimated time to reach 90%
          const incrementalProgress = (timeSinceThreeSeconds / estimatedTotalTime) * 40; // 40% from 50% to 90%
          newProgress = 50 + incrementalProgress;
          if (newProgress > 90) newProgress = 90;
        }
      } else {
        if (newProgress < 90) {
          newProgress = 90;
        }
        const timeSinceApiCompletion = now - apiCompletionTime.current;
        const completionProgress = (timeSinceApiCompletion / 2000) * 10; // 10% from 90% to 100%
        newProgress = 90 + completionProgress;
        if (newProgress >= 100) {
          newProgress = 100;
        }
      }

      setProgress(newProgress);

      // Update messages based on newProgress
      const progressThresholds = [0, 5, 15, 25, 35, 45, 55, 65, 75, 85, 90, 91];
      let newMessageIndex = 0;
      for (let i = progressThresholds.length - 1; i >= 0; i--) {
        if (newProgress >= progressThresholds[i]) {
          newMessageIndex = i;
          break;
        }
      }
      if (currentMessageIndex !== newMessageIndex) {
        setCurrentMessageIndex(newMessageIndex);
      }
    };

    const progressInterval = setInterval(updateProgress, 100);

    axios
      .post(`${baseUrl}/predict`, { image_url: imageUrl })
      .then((response) => {
        apiCompleted.current = true;
        apiCompletionTime.current = Date.now();
        setApiResponseData(response.data);
      })
      .catch((error) => {
        console.error('Error making prediction:', error);
        clearInterval(progressInterval);
        navigate('/select-image', {
          state: { error: 'Failed to analyze the image. Please try again.' },
        });
      });

    return () => {
      clearInterval(progressInterval);
    };
  }, [imageUrl, navigate]);

  useEffect(() => {
    if (progress >= 100 && apiCompleted.current) {
      navigate('/predictions', {
        state: { result: apiResponseData, image_url: imageUrl },
      });
    }
  }, [progress, navigate, apiResponseData, imageUrl]);

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Animated background (same as your existing background code) */}
      <div className="fixed inset-0 bg-black">
        {/* Background gradients and animations */}
        {/* ... (keep your existing background elements) */}
      </div>

      {/* Content */}
      <div className="relative min-h-screen backdrop-blur-xl flex flex-col items-center justify-center px-4">
        <div className="max-w-3xl w-full text-center">
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 animate-gradient-text bg-clip-text text-transparent mb-12">
            Processing...
          </h1>

          {/* Progress Bar */}
          <div className="w-full bg-gray-700 rounded-full h-12 mb-8 overflow-hidden">
            <div
              className="bg-blue-500 h-full transition-all duration-100 ease-out"
              style={{ width: `${Math.min(progress, 100)}%` }}
            />
          </div>

          {/* Changing Messages */}
          <div className="mb-12 h-8 flex items-center justify-center">
            <p className="text-xl text-gray-300">{messages[currentMessageIndex]}</p>
          </div>

          {/* Stylish Loading Spinner */}
          <div className="flex items-center justify-center">
            <div className="animate-spin-slow w-28 h-28 border-8 border-t-transparent border-b-transparent border-l-blue-400 border-r-purple-400 rounded-full"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressPage;
