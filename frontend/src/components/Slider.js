import React from 'react';

const Slider = ({ name, value, onChange, label }) => {
  const percentage = ((value - 1) / 4) * 100;

  return (
    <div className="slider-group">
      <label htmlFor={`${name}-slider`}>{label}</label>
      <div className="slider-wrapper">
        <input
          type="range"
          id={`${name}-slider`}
          name={name}
          min="1"
          max="5"
          value={value}
          onChange={onChange}
          className="custom-slider"
          style={{ '--fill-percent': `${percentage}%` }}
        />
        <span className="slider-value">{value}</span>
      </div>
    </div>
  );
};

export default Slider; 