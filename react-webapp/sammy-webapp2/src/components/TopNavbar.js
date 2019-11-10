import React from "react";
import "../styles/App.css";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav"
import Form from "react-bootstrap/Form"
import Image from 'react-bootstrap/Image'

export default function TopNavbar(props) {
  return (
    <Navbar fixed="top" bg="light" variant="light">
      <Image thumbnail style={{height:60,width:60}} src="https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/challenge_photos/000/882/690/datas/full_width.png"></Image>
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
