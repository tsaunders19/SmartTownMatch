import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="App-footer">
      <div className="footer-container">
        <div className="footer-column">
          <h3>Project</h3>
          <ul>
            <li>
              <a
                href="https://sites.google.com/view/smarttownmatch-research"
                target="_blank"
                rel="noopener noreferrer"
              >
                Information
              </a>
            </li>
          </ul>
        </div>
        <div className="footer-column">
          <h3>Resources</h3>
          <ul>
            <li>
              <a
                href="https://github.com/tsaunders19/SmartTownMatch"
                target="_blank"
                rel="noopener noreferrer"
              >
                GitHub
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div className="footer-copy">
        <p>Made by An Phan, Tiffany Saunders, and Palanivel Sathiya Moorthy for CS 539.</p>
      </div>
    </footer>
  );
};

export default Footer; 