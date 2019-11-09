import React, { useState } from 'react';
import "../styles/App.css";
import Jumbotron from 'react-bootstrap/Jumbotron'
import Container from 'react-bootstrap/Container'
import "../styles/App.css";
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'

export default function LoginForm(){
    return(
        <Row>
            <Col>
                <Jumbotron fluid>
                    <Container fluid="true">
                        <div className="Docs">
                            <h1>Documentation Goes Here!</h1>
                        </div>
                    </Container>
                </Jumbotron>
            </Col>
        </Row>
    )
}