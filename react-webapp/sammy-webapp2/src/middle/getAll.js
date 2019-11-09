const request = require('request'); 

export default function getAll() {
    request('http://40.76.48.13/ingredients', function ( error, response, body) {
        if (error){
            alert("Error, Cannot reach server")
        }
        console.log(body)
    })  
}

