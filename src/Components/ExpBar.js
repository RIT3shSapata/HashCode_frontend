import React, { useEffect, useState } from 'react';
import './ExpBar.css';

function ExpBar() {
  const [xp, setXP] = useState(0);
  const [pcount, setPcount] = useState(30);
  const [lastSeen, setLastSeen] = useState(
    new Date('Fri Jan 23 2021 08:00:10 GMT+0521')
  );

  useEffect(() => {
    let cdate = new Date();
    // {console.log((cdate - this.state.lastSeen)/(60*60*1000))}
    let diff = (cdate - lastSeen) / (60 * 60 * 1000);
    console.log(diff);
    if (diff > 0) {
      while (diff > 1 && pcount > 10) {
        diff = diff - 1;
        setPcount(pcount - 1);
        // this.setState((prevState) => { return {
        //     pcount : prevState.pcount - 1
        // }})
      }
    }
    let e = 2.7182828;
    setXP(10 * (Math.pow(e, 0.024 * pcount) - 1));
    // this.setState((prevState)=> {
    //     return {
    //     xp : 10*(Math.pow(e,0.024*(prevState.pcount)) - 1)
    //     }
    // })
    setInterval(() => {
      setXP(10 * (Math.pow(e, 0.024 * (pcount + 1)) - 1));
      setPcount(pcount + 1);
      // this.setState((prevState)=> {
      //     return {pcount: prevState.pcount + 1,
      //     xp : 10*(Math.pow(e,0.024*(prevState.pcount+1)) - 1)
      // }
      // })
    }, 60000);
  }, []);
  return (
    <div className="Exp__Bar">
      <h2 className="XP__Heading">XP</h2>
      <progress id='file' value={xp} max='100'>
        {xp}
      </progress>
      <span>{xp}</span>
      <br></br>
      {pcount}
    </div>
  );
}

export default ExpBar;
