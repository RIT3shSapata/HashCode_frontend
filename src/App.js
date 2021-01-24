import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';
import Sidenav from './Components/Sidenav';
import Logo from './Components/Logo';
import Resource from './Components/Resources';
import Problems from './Components/Problems';
import ProfileStudent from './Components/ProfileStudent';
import Code from './Views/Code';
import Friends from './Components/Friends';
import ExpBar from './Components/ExpBar';

function App() {
  return (
    <Router>
      <Switch>
        <Route path='/code'>
          <Code />
        </Route>
        <Route path='/problem'>
          <Logo />
          <ExpBar />
          <Sidenav />
          <Problems />
        </Route>
        <Route exact path='/'>
          <Logo />
          <ExpBar />
          <Sidenav />
          <ProfileStudent />
        </Route>
        <Route path='/connect'>
          <Logo />
          <ExpBar />
          <Sidenav />
          <Friends />
        </Route>
        <Route path='/resources'>
          <Logo />
          <ExpBar />
          <Sidenav />
          <Resource />
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
