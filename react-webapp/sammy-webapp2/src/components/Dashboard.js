import React, { useState } from 'react';
import "../styles/App.css";
import Jumbotron from 'react-bootstrap/Jumbotron'
import Container from 'react-bootstrap/Container'
import Button from 'react-bootstrap/Button'

export default function Dashboard(props) {
    return (
    <div className='StepBox'>
        <Jumbotron fluid>
            <Container fluid="true">
                <h1>Recipe Name</h1>
                <h1>1.</h1>
                <h1>2.</h1>
                <h1>3.</h1>
                <h1>4.</h1>
                <br></br>
            </Container>
        </Jumbotron>
        <Button variant="success" size="lg" block>
            Different Recipe!
        </Button>
    </div>
    )
  }
