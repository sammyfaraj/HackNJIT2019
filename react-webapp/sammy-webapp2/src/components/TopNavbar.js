import React from "react";
import "../styles/App.css";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav"
import Form from "react-bootstrap/Form"
import Image from 'react-bootstrap/Image'

export default function TopNavbar(props) {
  return (
    <div className="navbar">
    <Navbar fixed="top" bg="light" variant="light">

      <Nav className="mr-auto">
      {/* <Image thumbnail style={{height:80,width:80}} src="./Flip.gif"></Image>  */}
        <Navbar.Brand href="#dashhboard" onClick={() => props.setDocumentation(false)}><h1 style={{marginLeft: .5 + 'em'}} >InstaCook</h1></Navbar.Brand>
      </Nav>
      
      <Form inline>
        <Nav.Link href="#dashhboard" onClick={() => props.setDocumentation(false)}>Home</Nav.Link>
        <Nav.Link href="#documentation" onClick={() => props.setDocumentation(true)}>Documentation</Nav.Link>
      </Form>
    </Navbar>
    </div>
  );
}
