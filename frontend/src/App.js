import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import LoadingScreen from './components/LoadingScreen';
import Header from './components/Header';
import PreferencesForm from './components/PreferencesForm';
import Results from './components/Results';

function App() {
  const [backendReady, setBackendReady] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [cluster, setCluster] = useState(() => {
    return sessionStorage.getItem('selectedCluster') || 'Suburb';
  });
  const [weights, setWeights] = useState({
    affordability: 3,
    safety: 3,
    education: 3,
    amenities: 3,
    walkability: 3,
    transit: 3,
    bike: 3
  });

  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    sessionStorage.setItem('selectedCluster', cluster);
  }, [cluster]);

  useEffect(() => {
    const checkBackend = async () => {
      try {
        await axios.get('/api/health');
        setBackendReady(true);
      } catch (e) {
        console.error("Backend not ready, retrying...", e);
        setTimeout(checkBackend, 3000);
      }
    };
    checkBackend();
  }, []);

  useEffect(() => {
    document.body.className = isDarkMode ? 'dark-mode' : 'light-mode';
  }, [isDarkMode]);

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setWeights(prevWeights => ({
     ...prevWeights,
      [name]: parseInt(value, 10)
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');
    setRecommendations([]);

    try {
      const requestPayload = {
        cluster: cluster,
        weights: {
          MedianHomePrice: weights.affordability,
          SafetyScore: weights.safety,
          EducationScore: weights.education,
          AmenitiesScore: weights.amenities,
          WalkScore: weights.walkability,
          TransitScore: weights.transit,
          BikeScore: weights.bike
        }
      };

      const response = await axios.post('/api/recommendations', requestPayload);
      if (Array.isArray(response.data)) {
        setRecommendations(response.data);
      } else {
        setError('Unexpected response from server.');
        console.error('API response not array:', response.data);
      }

    } catch (err) {
      setError('Failed to fetch recommendations. Please ensure the backend server is running.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  if (!backendReady) {
    return <LoadingScreen />;
  }

  return (
    <div className="App">
      <Header
          isDarkMode={isDarkMode}
          onToggleDarkMode={() => setIsDarkMode(!isDarkMode)}
      />
      <main className="App-main">
        <aside className={`sidebar ${isSidebarOpen ? 'open' : 'collapsed'}`}> 
          {isSidebarOpen && (
            <button
              type="button"
              className="sidebar-handle"
              onClick={() => setIsSidebarOpen(false)}
            >
              ❮
            </button>
          )}
          <PreferencesForm
            cluster={cluster}
            setCluster={setCluster}
            weights={weights}
            handleSliderChange={handleSliderChange}
            handleSubmit={handleSubmit}
            isLoading={isLoading}
          />
        </aside>

        {!isSidebarOpen && (
          <button
            type="button"
            className="sidebar-expand"
            onClick={() => setIsSidebarOpen(true)}
          >
            ❯
          </button>
        )}

        <section className="content">
          <Results recommendations={recommendations} error={error} />
        </section>
      </main>
      <footer className="App-footer">
        <p>Made by An Phan, Tiffany Saunders and Palanivel Sathiya Moorthy for CS 539</p>
      </footer>
    </div>
  );
}

export default App;