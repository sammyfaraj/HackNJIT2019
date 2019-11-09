import React from "react";
import "../styles/App.css";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav"
import Form from "react-bootstrap/Form"

export default function TopNavbar(props) {
  return (
    <Navbar fixed="top" bg="light" variant="light">
      <Nav className="mr-auto">
        <Navbar.Brand href="#dashhboard" onClick={() => props.setDocumentation(false)}><h2>The Fun Cooker</h2></Navbar.Brand>
      </Nav>
      <Form inline>
        <Nav.Link href="#dashhboard" onClick={() => props.setDocumentation(false)}>Home</Nav.Link>
        <Nav.Link href="#documentation" onClick={() => props.setDocumentation(true)}>Documentation</Nav.Link>
      </Form>
    </Navbar>
  );
}
