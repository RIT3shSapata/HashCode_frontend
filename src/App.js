import { BrowserRouter as Router, Route, Switch } from 'react-router-dom';
import './App.css';
import Code from './Views/Code';

function App() {
  return (
    <Router>
      <Switch>
        <Route path='/code'>
          <Code />
        </Route>
        <Route path='/'>
          <h1>reLearn Frontend</h1>
        </Route>
      </Switch>
    </Router>
  );
}

export default App;
