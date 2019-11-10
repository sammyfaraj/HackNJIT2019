import React from "react";
import "../styles/App.css";
import Button from 'react-bootstrap/Button'
import {getRecipe} from '../middle/getAll'

export default function TopNavbar(props) {
  return (
    <Button onClick={() => getRecipe(props.selectedIngredients,props.setRecipe)} variant="success" size="lg" block>
        Get Your Recipe!
    </Button>
  );
}
