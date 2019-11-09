import React, { useState } from 'react';
import "../styles/App.css";
import Card from "react-bootstrap/Card"
import getAll from '../middle/getAll'
import IngredientSelect from './IngredientSelect'

export default function CodeSnippetList(props) {
  const [masterIngredients, setMaster] = useState([]);

  getAll(setMaster)

  let Snippets = 
  [
    {
      Name:"Master",
      Description:"Search all Ingredients!",
      Route:"facialrecognition",
      Code:"",
    },
    {
      Name:"Protein",
      Description:"Select your Proteins!",
      Route:"facialrecognition",
      Code:"",
    },
    {
      Name:"Dairy",
      Description:"Select your Dairy!",
      Route:"videoupload",
      Code:"",
    },
    {
      Name:"Grains",
      Description:"Select your Grains!",
      Route:"snippet1",
      Code:"",
    },
    {
      Name:"Fruits",
      Description:"Select your Fruits!",
      Route:"snippet2",
      Code:""
    },
  ];

  return (
    Snippets.map((snippet) =>
    <div className="IngredientCard">
      <Card>
          <h3>{snippet.Name}</h3>
          <p>{snippet.Description}</p>
        <IngredientSelect options={masterIngredients}></IngredientSelect>
        <br></br>
      </Card>
      <br></br>
    </div>
    )
  )
}

//onClick={() => APIrequest(props.setCheckpoints, props.setComputation, props.setLatency, snippet.Route)} as={Card.Header}