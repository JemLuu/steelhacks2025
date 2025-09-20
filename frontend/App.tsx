import React, { useState } from 'react';
import LandingPage from './components/LandingPage';
import AnalysisScreen from './components/AnalysisScreen';

export default function App() {
  // Main application component
  const [currentScreen, setCurrentScreen] = useState<'landing' | 'analysis'>('landing');
  const [redditHandle, setRedditHandle] = useState('');

  const handleAnalyze = (handle: string) => {
    setRedditHandle(handle);
    setCurrentScreen('analysis');
  };

  const handleBackToLanding = () => {
    setCurrentScreen('landing');
    setRedditHandle('');
  };

  return (
    <div className="min-h-screen">
      {currentScreen === 'landing' ? (
        <LandingPage onAnalyze={handleAnalyze} />
      ) : (
        <AnalysisScreen
          redditHandle={redditHandle}
          onBack={handleBackToLanding}
        />
      )}
    </div>
  );
}