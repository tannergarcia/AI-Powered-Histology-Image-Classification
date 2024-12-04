import React from 'react';
import { Route, Routes } from 'react-router-dom';
import WelcomePage from './components/WelcomePage';
import UploadImagePage from './components/UploadImagePage';
import ProgressPage from './components/ProgressPage';
import SelectImagePage from './components/SelectImagePage';
import PredictionsPage from './components/PredictionsPage';
import ResultsPage from './components/ResultsPage';


function App() {
  return (
    <Routes>
      <Route path="/" element={<WelcomePage />} />
      <Route path="/upload" element={<UploadImagePage />} />
      <Route path="/progress" element={<ProgressPage />} />
      <Route path="/select-image" element={<SelectImagePage />} />
      <Route path="/predictions" element={<PredictionsPage />} />
      <Route path="/results" element={<ResultsPage />} />
    </Routes>
  );
}

export default App;
