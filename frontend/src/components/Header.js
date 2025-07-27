import React from 'react';
import { FiSun, FiMoon } from 'react-icons/fi';

const Header = ({ isDarkMode, onToggleDarkMode }) => {
  return (
    <header className="app-header">
      <div className="logo">
        <h1>SmartTownMatch</h1>
        <p>Find Your Perfect Massachusetts Town</p>
      </div>
      <div className="theme-toggle">
        <button onClick={onToggleDarkMode}>
          {isDarkMode ? <FiSun /> : <FiMoon />}
        </button>
      </div>
    </header>
  );
};

export default Header; 