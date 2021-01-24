import React, { useState } from "react";
import Button from "@material-ui/core/Button";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import CardActions from "@material-ui/core/CardActions";
import IconButton from "@material-ui/core/IconButton";
import Typography from "@material-ui/core/Typography";
import FavoriteIcon from "@material-ui/icons/Favorite";
import { Link } from 'react-router-dom';
import './QuestionCard.css';

export default function QuestionCard(props) {
  return (
      //<div className="Question__Bar">
          //  <p className="Question">{props.question}</p>
           // <p className="Difficulty">{props.difficulty}</p>
   //   </div>

    <Card className="Question__Bar">
    <CardContent>
    <Typography variant="body2" color="textSecondary" component="p" className='Question'>
        {props.question}
    </Typography>
    <Typography variant="body2" color="textSecondary" component="p" className='Difficulty'>
        {props.difficulty}
    </Typography>
    </CardContent>
    <Link to='/code'>
    <Button className="Question__Button" variant="contained" color="#AACDAE">
          SOLVE
        </Button>
        </Link>
    </Card>
    );
}
