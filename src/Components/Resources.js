import React, { useState } from 'react';
import Card from './Card';
import './Resources.css';

const lists = [
  {
    image: 'https://dummyimage.com/600x400/000/fff',
    content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.1`,
  },
  {
    image: 'https://dummyimage.com/600x400/000/fff',
    content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.2`,
  },
  {
    image: 'https://dummyimage.com/600x400/000/fff',
    content: `This impressive paella is a perfect party dish and a fun meal to cook
    together with your guests. Add 1 cup of frozen peas along with the
    mussels, if you like.3`,
  },
];

// function Resources() {
//   return (
//       <div>
//           <h2 className="Resource__Heading">RESOURCE</h2>
//       <div className="Resource" style={{position: "relative", left:"500px", top:"145px"}}>
//             <div style={{ display: "flex", flexFlow: "row wrap", margin: "0 5px" }}>
//             {lists.map((list, idx) => {
//             return <Card key={idx} image={list.image} content={list.content} />;
//             })}
//         </div>
//       </div>
//       </div>
//     );
// }

function Resources() {
  const [query, setQuery] = useState('');
  const [articles, setArticles] = useState([]);

  const handlequery = (e) => {
    setQuery(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const requestOptions = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title: query }),
    };
    console.log(query);
    fetch('http://127.0.0.1:5000/articles', requestOptions)
      .then((response) => response.json())
      .then((res) => {
        setArticles(res.arts);
      });
  };

  return (
    <div className='Resource__Query'>
      <form onSubmit={handleSubmit}>
        <h2 className='Resource__Heading'>RESOURCE</h2>
        <input
          type='text'
          className='form-control'
          placeholder='Search for resources'
          id='xyz'
          name='studentname'
          value={query}
          onChange={handlequery}
          required
        />
        <input type='submit' className='Query__Button' value='Submit' />
      </form>
      <div>
        {articles.map((x) => {
          return (
            <p>
              <a href={x} target='_blank'>
                {x}
              </a>
            </p>
          );
        })}
      </div>
    </div>
  );
}

export default Resources;
