import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';

import Sidenav from './Components/Sidenav';
import Logo from './Components/Logo';
import Resource from './Components/Resources';
import Problems from './Components/Problems';
import Code from './Views/Code';


function App() {
  return (
    <Router>
      <Switch>
        <Route path='/code'>
          <Code />
        </Route>
        <Route path='/'>
         <Logo />
      <Sidenav />
      <Problems /> 
        </Route>
      </Switch>
    </Router>
)}

export default App;
