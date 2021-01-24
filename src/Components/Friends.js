import React from 'react';
import FriendCard from './FriendCard';
import './Friends.css';

const lists = [
  {
    rec: 'Task1',
   
  },
  {
      rec: 'Task2',
     
  }
];

function Friends() {
  return (
    <div>
    <h1 className='Friends__Heading'>RECOMMENDED</h1>
    <div className="Problem" style={{position: "relative", left:"500px", top:"125px"}}>
          <div style={{ display: "flex", flexFlow: "row wrap", margin: "0 5px" }}>
          {lists.map((list, idx) => {
          return <FriendCard key={idx} rec={list.rec} />;
          })}
      </div>
    </div>
  </div>
  );
}

export default Friends;
