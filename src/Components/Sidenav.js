import React from 'react';
import DashboardIcon from '@material-ui/icons/Dashboard';
import LaptopChromebookIcon from '@material-ui/icons/LaptopChromebook';
import CodeIcon from '@material-ui/icons/Code';
import PeopleIcon from '@material-ui/icons/People';
import PowerSettingsNewIcon from '@material-ui/icons/PowerSettingsNew';
import { Link } from 'react-router-dom';
import profile from '../img/profile.jpg'
import './Sidenav.css';

function Sidenav() {
  return (
    <div className='Nav'>
        <img className='Profile__Img' src={profile} alt="Profile" />
        <p className='User__Name'>User01</p>
        <div className='Sidebar__Link'>
            <div className='Sidebar__Pages'>
                <DashboardIcon className='Navbar__dashboard'/>
                <p className='Page'>PROFILE</p>
            </div>
          

            <div className='Sidebar__Pages'>
                <LaptopChromebookIcon className='Navbar__resources' />
                <p className='Page'>RESOURCES</p>
            </div>
           

            <div className='Sidebar__Pages'>
                <CodeIcon className='Navbar__code' />
                <p className='Page'>CODER</p>
            </div>
            

            <div className='Sidebar__Pages'>
                <PeopleIcon className='Navbar__connect' />
                <p className='Page'>CONNECT</p>
            </div>
        

            <div className='Sidebar__Pages'>
                <PowerSettingsNewIcon className='Navbar__logout' />
                <p className='Page'>LOGOUT</p>
            </div>
    </div>

    </div>
  );
}

export default Sidenav;
