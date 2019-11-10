import React, { useState } from 'react';
import "../styles/App.css";
import "bootstrap/dist/css/bootstrap.min.css";
import TopNavbar from "./TopNavbar";
import Container from 'react-bootstrap/Container'
import Row from 'react-bootstrap/Row'
import Col from 'react-bootstrap/Col'
import CodeSnippetList from './CodeSnippetList'
import Dashboard from './Dashboard'
import Documentation from '../docs/documentation'
import SubmitButton from './SubmitButton.js'

function App() {
  const [documentation, setDocumentation] = useState(false);
  return (
    <div className="App">
      {!documentation &&
      <div className="Dashboard">
      <TopNavbar setDocumentation={setDocumentation}></TopNavbar>
        <Container fluid="true">
          <Row>
            <Col xs={3}>
              <CodeSnippetList></CodeSnippetList>
              <SubmitButton></SubmitButton>
            </Col>
            <Col xs ={9}>
              <Dashboard></Dashboard>
            </Col>
          </Row>
        </Container>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
        <br></br>
      </div>}
      {documentation &&
        <div className="Dashboard">
          <TopNavbar setDocumentation={setDocumentation}></TopNavbar>
          <Documentation></Documentation>
        </div>
      }
    </div>
    )
}

export default App;
