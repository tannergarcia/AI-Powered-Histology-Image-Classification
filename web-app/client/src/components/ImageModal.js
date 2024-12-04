// src/components/ImageModal.js
import React, { useEffect } from 'react';

const ImageModal = ({ isOpen, onClose, imageUrl, altText }) => {
  useEffect(() => {
    const handleEsc = (event) => {
      if (event.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleEsc);
    }
    return () => window.removeEventListener('keydown', handleEsc);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-75 backdrop-blur-sm transition-opacity duration-300"
      onClick={onClose}
    >
      <div
        className="relative max-w-4xl w-full max-h-full p-4"
        onClick={(e) => e.stopPropagation()} // Prevents closing when clicking inside the modal
      >
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-white text-3xl hover:text-gray-300 focus:outline-none"
          aria-label="Close"
        >
          &times;
        </button>
        <img
          src={imageUrl}
          alt={altText}
          className="w-full h-auto max-h-screen rounded-lg shadow-xl"
        />
      </div>
    </div>
  );
};

export default ImageModal;
