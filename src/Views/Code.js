import React from 'react';
import Editor from '../Components/Editor/Editor';

function Code() {
  return (
    <div className='code'>
      <div className='code__left'>
        <h1>Question</h1>
      </div>
      <div className='code__right'>
        <Editor />
      </div>
    </div>
  );
}

export default Code;
