import React, { useState } from 'react';
import "../styles/App.css";
import Jumbotron from 'react-bootstrap/Jumbotron'
import Container from 'react-bootstrap/Container'
import ListGroup from 'react-bootstrap/ListGroup'
import SubmitButton from './SubmitButton.js'

export default function Dashboard(props) {
    var newlist = []

    props.selectedIngredients.forEach(function(element){
        newlist.push(element.label)
     });

    return (
    <div className='RightBox'>
        <Jumbotron fluid>
            <div className="StepBox">
            <Container fluid="true" >
                <h1>Your Recipe!</h1>
                <br></br>
                <h3>Ingredients: {newlist.toString()}</h3>
                <br></br>
                <h3>Steps:</h3>
                <ListGroup variant="flush">
                {props.recipe.map((step) =>
                    <ListGroup.Item>{step}</ListGroup.Item>)}
                </ListGroup>
               
            </Container>
            </div>
        </Jumbotron>
        <SubmitButton selectedIngredients={props.selectedIngredients} setRecipe={props.setRecipe}></SubmitButton>
    </div>
    )
  }
