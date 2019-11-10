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

function App() {
  const [documentation, setDocumentation] = useState(false);
  const [selectedIngredients, setSelected] = useState([]);
  const [recipe,setRecipe] = useState(["1.","2.","3.","4.","5."]);

  if (!selectedIngredients){
    setSelected([])
  }
  return (
    <div className="App">
      {!documentation &&
      <div className="Dashboard">
      <TopNavbar setDocumentation={setDocumentation}></TopNavbar>
        <Container fluid="true">
          <Row>
            <Col xs={3}>
              <CodeSnippetList setSelected={setSelected}></CodeSnippetList>
            </Col>
            <Col xs ={9}>
              <Dashboard selectedIngredients={selectedIngredients} setRecipe={setRecipe} recipe={recipe}></Dashboard>
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
