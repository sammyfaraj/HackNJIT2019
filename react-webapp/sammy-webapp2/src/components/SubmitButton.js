import React from "react";
import "../styles/App.css";
import Button from 'react-bootstrap/Button'
import getAll from '../middle/getAll'

export default function TopNavbar(props) {
  return (
    <Button onClick={() => getAll()} variant="success" size="lg" block>
        Get Your Recipe!
    </Button>
  );
}
