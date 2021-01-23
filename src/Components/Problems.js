import React from 'react';
import Card from './QuestionCard'
import './Problems.css';

const lists = [
    {
      question: 'Task1',
      difficulty: 'Easy'
    },
    {
        question: 'Task2',
        difficulty: 'Hard'
    }
  ];

function Problems() {
  return (
      <div>
          <h2 className="Problems__Heading">IMPROVE SKILLS</h2>
      <div className="Problem" style={{position: "relative", left:"500px", top:"125px"}}>
            <div style={{ display: "flex", flexFlow: "row wrap", margin: "0 5px" }}>
            {lists.map((list, idx) => {
            return <Card key={idx} question={list.question} difficulty={list.difficulty} />;
            })}
        </div>
      </div>
      </div>
    );
}

export default Problems;