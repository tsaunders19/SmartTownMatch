import React from 'react';
import Slider from './Slider';
import LifestyleSelect from './LifestyleSelect';

const PreferencesForm = ({ cluster, setCluster, weights, handleSliderChange, handleSubmit, isLoading }) => {
  return (
    <form onSubmit={handleSubmit} className="preferences-form">
      <div className="form-section">
        <h2 className="form-section-title">1. Choose Your Ideal Lifestyle</h2>
        <p className="form-section-description">
          Select the type of community that best fits your daily life.
        </p>
        <div className="lifestyle-dropdown">
          <LifestyleSelect value={cluster} onChange={setCluster} />
        </div>
      </div>

      <div className="form-section">
        <h2 className="form-section-title">2. What Matters Most to You?</h2>
        <p className="form-section-description">
          Adjust the sliders to rank your priorities from 1 (not important) to 5 (very important).
        </p>
        <div className="sliders-container">
          {Object.keys(weights).map(weightName => (
            <Slider
              key={weightName}
              name={weightName}
              label={weightName.charAt(0).toUpperCase() + weightName.slice(1)}
              value={weights[weightName]}
              onChange={handleSliderChange}
            />
          ))}
        </div>
      </div>

      <button type="submit" className="submit-btn" disabled={isLoading}>
        {isLoading ? 'Searching...' : 'Find My Perfect Town'}
      </button>
    </form>
  );
};

export default PreferencesForm; 