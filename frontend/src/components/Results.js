import React from 'react';
import { motion } from 'framer-motion';

const Score = ({ label, value, imputed, low, high }) => {
  if (value === null || value === undefined) return null;
  
  const displayValue = Math.round(value);

  return (
    <p className="score-item">
      <strong>{label}:</strong> {displayValue}/100
      {imputed && <span title="This score was estimated due to missing data." className="imputed-icon"> ⚠️</span>}
      {low !== undefined && high !== undefined && (
        <span className="confidence-interval"> ({Math.round(low)} - {Math.round(high)})</span>
      )}
    </p>
  );
};

const Results = ({ recommendations, error }) => {
  if (error) {
    return <p className="error-message">{error}</p>;
  }

  if (recommendations.length === 0) {
    return null;
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1
    }
  };

  return (
    <motion.div 
      className="results-container"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <h2 className="results-title">Your Top Recommendations</h2>
      <motion.ul className="recommendations-list" variants={containerVariants}>
        {recommendations.map((town) => (
          <motion.li 
            key={town.TownName} 
            className="recommendation-item"
            variants={itemVariants}
          >
            <h3 className="town-header">{town.TownName}</h3>
            <div className="town-details">
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
            </div>
          </motion.li>
        ))}
      </motion.ul>
    </motion.div>
  );
};

export default Results; 