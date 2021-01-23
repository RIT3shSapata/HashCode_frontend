import React, { useState } from "react";
import Button from "@material-ui/core/Button";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import CardActions from "@material-ui/core/CardActions";
import IconButton from "@material-ui/core/IconButton";
import Typography from "@material-ui/core/Typography";
import FavoriteIcon from "@material-ui/icons/Favorite";

export default function ListCard(props) {
  return (
    <Card style={{ maxWidth: "300px", margin: "0 5px"}}>
      <img style={{ width: "100%" }} src={props.image} alt="image-card" />
      <CardContent>
        <Typography variant="body2" color="textSecondary" component="p">
          {props.content}
        </Typography>
      </CardContent>
      <CardActions disableSpacing>
        <Button variant="contained" color="#AACDAE">
          GO TO SITE
        </Button>
      </CardActions>
    </Card>
  );
}
