import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const Score = ({ label, value, imputed, low, high }) => {
  if (value === null || value === undefined) return null;
  
  const displayValue = Math.round(value);

  return (
    <p>
      <strong>{label}:</strong> {displayValue}/100
      {imputed && <span title="This score was estimated due to missing data." className="imputed-icon"> ⚠️</span>}
      {low !== undefined && high !== undefined && (
        <span className="confidence-interval"> ({Math.round(low)} - {Math.round(high)})</span>
      )}
    </p>
  );
};


function App() {
  const [cluster, setCluster] = useState('Suburb');
  const [weights, setWeights] = useState({
    affordability: 3, // MedianHomePrice
    safety: 3,        // SafetyScore
    education: 3,     // EducationScore
    amenities: 3,     // AmenitiesScore
    walkability: 3,   // WalkScore
    transit: 3,       // TransitScore
    bike: 3           // BikeScore
  });

  const [recommendations, setRecommendations] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

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

  return (
    <div className="App">
      <header className="App-header">
        <h1>SmartTownMatch</h1>
        <p>Find Your Perfect Massachusetts Town</p>
      </header>
      <main>
        <form onSubmit={handleSubmit} className="preferences-form">
          <h2>Your Preferences</h2>
          
          <div className="form-group">
            <label htmlFor="cluster-select">1. Choose a lifestyle:</label>
            <select id="cluster-select" value={cluster} style={{fontSize: 16}} onChange={e => setCluster(e.target.value)}>
              <option value="Suburb">Suburb</option>
              <option value="City">City</option>
              <option value="Rural">Rural</option>
            </select>
          </div>

          <p>2. Rank what's important to you (1 = Not important, 5 = Very important):</p>
          
          <div className="sliders-container">
            {Object.keys(weights).map(weightName => (
              <div className="form-group slider-group" key={weightName}>
                <label htmlFor={`${weightName}-slider`}>{weightName.charAt(0).toUpperCase() + weightName.slice(1)}</label>
                <input
                  type="range"
                  id={`${weightName}-slider`}
                  name={weightName}
                  min="1"
                  max="5"
                  value={weights[weightName]}
                  onChange={handleSliderChange}
                />
                <span>{weights[weightName]}</span>
              </div>
            ))}
          </div>

          <button type="submit" disabled={isLoading}>
            {isLoading? 'Finding Towns...' : 'Find My Town'}
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}

        <div className="results-container">
          {recommendations.length > 0 && <h2>Top Recommendations</h2>}
          <ul className="recommendations-list">
            {Array.isArray(recommendations) && recommendations.map((town) => (
              <li key={town.TownName} className="recommendation-item">
                <h3>{town.TownName}</h3>
                <Score label="Match Score" value={town.MatchScore} low={town.MatchScoreLow} high={town.MatchScoreHigh} />
                {town.County && <p><strong>County:</strong> {town.County}</p>}
                <p><strong>Lifestyle:</strong> {town.ClusterLabel}</p>
                <p><strong>Median Home Price:</strong> ${new Intl.NumberFormat().format(town.MedianHomePrice)}</p>
                <p><strong>Population:</strong> {new Intl.NumberFormat().format(town.Population)}</p>
                <Score label="Safety Score" value={town.SafetyScore_norm * 100} imputed={town.SafetyScoreImputed} />
                <Score label="Education Score" value={town.EducationScore_norm * 100} imputed={town.EducationScoreImputed} />
                <Score label="Walk Score®" value={town.WalkScore} />
                <Score label="Transit Score" value={town.TransitScore} />
                <Score label="Bike Score" value={town.BikeScore} />
              </li>
            ))}
          </ul>
        </div>
      </main>
      <footer className="App-footer">
        <p>Made by An Phan, Tiffany Saunders and Palanivel Sathiya Moorthy for CS 539</p>
      </footer>
    </div>
  );
}

export default App;