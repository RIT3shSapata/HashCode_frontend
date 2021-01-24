import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActionArea from '@material-ui/core/CardActionArea';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import CardMedia from '@material-ui/core/CardMedia';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import GitHubIcon from '@material-ui/icons/GitHub';

const useStyles = makeStyles({
  root: {
    maxWidth: 270,
    borderRadius: 10,
    marginTop: 15
  },
  media: {
    height: 200,
    width: '100%',
    objectFit: 'contain',
  },
});

function FriendCard(props) {
  const classes = useStyles();

  return (
    <div>
      <Card className={classes.root}>
        <CardActionArea>
          <CardMedia
            className={classes.media}
            image='https://preview.redd.it/7fcu41hi2re41.png?auto=webp&s=038dc69c1622eb809ebbfbc15c22c5f6739dca65'
            title='Friend'
          />
          <CardContent>
            
            <Typography variant='body2' color='textSecondary' component='p'>
              {props.rec}
            </Typography>
          </CardContent>
        </CardActionArea>
        <CardActions>
          <Button size='small' color='primary'>
            Add Friend
          </Button>
          <Button size='small' color='primary'>
            Learn More
          </Button>
        </CardActions>
        <GitHubIcon className='Github__Icon'/>
      </Card>
    </div>
  );
}

export default FriendCard;
