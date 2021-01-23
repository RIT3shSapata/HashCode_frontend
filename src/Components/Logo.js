import React from 'react';
import logo from '../img/logo.png'
import './Logo.css';

function Logo() {
  return (
      <div className="Logo__box">
            <img className='Logo' src={logo} alt="Logo" />
            <h2 className='Main__Heading'>reLearn</h2>
      </div>
    );
}

export default Logo;