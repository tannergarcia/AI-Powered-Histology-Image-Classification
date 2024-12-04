import React, { useEffect, useRef } from 'react';

const CircularProgress = ({ percentage, label }) => {
  const circleRef = useRef(null);
  const animationDuration = 1500;

  const strokeWidth = 10;
  const radius = 100 - strokeWidth / 2;
  const circumference = 2 * Math.PI * radius;

  useEffect(() => {
    const circle = circleRef.current;
    if (circle) {
      circle.style.transition = `stroke-dashoffset ${animationDuration}ms ease-in-out`;
      circle.style.strokeDashoffset =
        circumference - (percentage / 100) * circumference;
    }
  }, [percentage, circumference]);

  return (
    <div className="relative w-48 h-48 sm:w-64 sm:h-64 flex items-center justify-center">
      <div className="absolute inset-0 bg-white bg-opacity-10 backdrop-blur-md rounded-full shadow-2xl"></div>
      <svg
        className="transform -rotate-90"
        width="200"
        height="200"
        viewBox="0 0 200 200"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Background Circle */}
        <circle
          cx="100"
          cy="100"
          r={radius}
          stroke="rgba(255, 255, 255, 0.1)"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress Circle */}
        <circle
          ref={circleRef}
          cx="100"
          cy="100"
          r={radius}
          stroke="url(#gradient)"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={circumference}
          strokeLinecap="round"
          filter="url(#glow)"
        />
        {/* Gradient and Glow Definitions */}
        <defs>
          <linearGradient id="gradient" gradientTransform="rotate(90)">
            <stop offset="0%" stopColor="#3b82f6" />
            <stop offset="100%" stopColor="#9333ea" />
          </linearGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="4" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
      </svg>
      {/* Percentage and Label */}
      <div className="absolute flex flex-col items-center px-2 text-center">
        <span className="text-3xl font-semibold text-white tracking-tight font-sans">
          {percentage}%
        </span>
        {label && (
          <div className="text-sm text-gray-300 mt-1 font-sans leading-tight">
            {label}
          </div>
        )}
      </div>
    </div>
  );
};

export default CircularProgress;
