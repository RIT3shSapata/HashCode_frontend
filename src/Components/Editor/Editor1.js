import React, { useRef, useState, useEffect} from 'react';
import AceEditor from 'react-ace';
import server from '../../Server';
// import text from './competitions.txt'; // Relative path to your File

import 'ace-builds/src-noconflict/mode-python';
import 'ace-builds/src-noconflict/theme-cobalt';
import 'ace-builds/src-noconflict/ext-language_tools';

function Editor1() {
  const [code, setCode] = useState('');
  const codeRef = useRef('');
  let posted = new Date('Sun Jan 24 2021 01:00:00 GMT+0530');
  let scored = -10;
  let started = new Date('Sun Jan 24 2021 01:25:20 GMT+0530');
  let maxp = 10*Math.exp(-0.23 * (new Date('Sun Jan 24 2021 01:25:00 GMT+0530') - new Date('Sun Jan 24 2021 01:00:20 GMT+0530'))/(1000*60*60))
  console.log(maxp)
  const onChange = (newValue) => {
    setCode(newValue);
  };
  const submit = async () => {
    // setCode(codeRef.current.value);
    console.log(code);
    const data = {
      language: 'python3',
      script: code,
      stdin: 'TESTING',
    };
    try {
      const res = await server.post('/code', data);
      console.log(res.data);
    } catch (error) {
      console.log(error);
    }
    if(1) //check if answer's right
    {
      let timeTaken = (new Date('Sun Jan 24 2021 1:59:20 GMT+0530') - new Date('Sun Jan 24 2021 1:25:20 GMT+0530')) / (60 * 10 *10);
      scored = maxp - maxp*(timetaken/370); //make it as state variable
      console.log(scored)
    }
  };

  return (
    <div>
      <div>
        <p>Given an array of integers, Find the subarray with maximum sum!!!</p>

        <p>Max Points : {maxp}</p>
        <p>Posted on : Sun Jan 24 2021 01:00:00 GMT+0530</p>
        <p>Started : Sun Jan 24 2021 01:25:00 GMT+0530</p>
      </div>
      <AceEditor
        ref={codeRef}
        mode='python'
        theme='cobalt'
        onChange={onChange}
        value={code}
        name='UNIQUE_ID_OF_DIV'
        height='90px'
        width='90vw'
        editorProps={{ $blockScrolling: true }}
        setOptions={{
          enableBasicAutocompletion: true,
          enableLiveAutocompletion: true,
          enableSnippets: true,
        }}
      />
      <button onClick={submit}>Submit</button>
      <div>Gained Points:{scored}</div>
    </div>
  );
}

export default Editor1;
