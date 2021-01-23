import React from 'react';
import './Logo.css';
import Card from './Card'
import './Resources.css';

const lists = [
    {
      image: "https://dummyimage.com/600x400/000/fff",
      content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.1`
    },
    {
      image: "https://dummyimage.com/600x400/000/fff",
      content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.2`
    },
    {
      image: "https://dummyimage.com/600x400/000/fff",
      content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.3`
    }
  ];

function Logo() {
  return (
      <div>
          <h2 className="Resource__Heading">RESOURCE</h2>
      <div className="Resource" style={{position: "relative", left:"500px", top:"145px"}}>
            <div style={{ display: "flex", flexFlow: "row wrap", margin: "0 5px" }}>
            {lists.map((list, idx) => {
            return <Card key={idx} image={list.image} content={list.content} />;
            })}
        </div>
      </div>
      </div>
    );
}

export default Logo;