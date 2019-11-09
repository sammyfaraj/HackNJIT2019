const request = require('request'); 

export default function getAll(setter) {
    request('http://40.76.48.13/ingredients', function ( error, response, body) {
        if (error){
            alert("Error, Cannot reach server")
        }
        body = JSON.parse(body);
        let ingredients = body['ingredients'];
        const options = [];
        ingredients.forEach(function (item, index) {
            if (item.length < 20){ 
            options.push({
              label: item,
              value: index
            });
          }})
        setter(options)
    })  
}

