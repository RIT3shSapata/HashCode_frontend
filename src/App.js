import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';
import Sidenav from './Components/Sidenav';
import Logo from './Components/Logo';
import Resource from './Components/Resources';
import Problems from './Components/Problems';
import ProfileStudent from './Components/ProfileStudent'
import Code from './Views/Code';
import Friends from './Components/Friends';


function App() {
  return (
    <Router>
      <Switch>
        <Route path='/code'>
          <Code />
        </Route>
        <Route path='/problem'>
          <Logo />
          <Sidenav />
          <Problems /> 
        </Route>
        <Route path='/'>
          <Logo />
          <Sidenav />
          <ProfileStudent /> 
        </Route>
        <Route path='/connect'>
          <Logo />
          <Sidenav />
          <Friends /> 
        </Route>
        <Route path='/resource'>
          <Logo />
          <Sidenav />
          <Resource /> 
        </Route>
      </Switch>
    </Router>
)}

export default App;
