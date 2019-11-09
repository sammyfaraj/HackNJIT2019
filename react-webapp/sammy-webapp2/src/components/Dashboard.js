import React, { useState } from 'react';
import "../styles/App.css";
import Jumbotron from 'react-bootstrap/Jumbotron'
import Container from 'react-bootstrap/Container'

export default function Dashboard(props) {
    return (
     <Jumbotron fluid>
        <Container fluid="true">
            <h1>Recipe Name</h1>
            <br></br>
        </Container>
    </Jumbotron>
    )
  }
