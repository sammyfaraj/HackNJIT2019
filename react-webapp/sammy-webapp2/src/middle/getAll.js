const request = require('request'); 

export default function getAll(setter) {
    request('http://40.76.48.13/ingredients', function ( error, response, body) {
        body = JSON.parse(body);
        let ingredients = body['ingredients'];
        let proteins = body['proteins'];
        let vegetables = body['vegetables'];
        let grains = body['grains'];
        let dairy = body['dairy'];
        let fruits = body['fruits'];
        const options = {'ingredients':[],'proteins':[],'vegetables':[],'grains':[],'dairy':[],'fruits':[]};
        ingredients.forEach(function (item, index) {
            if (item.length < 12){ 
            options.ingredients.push({
              label: item,
              value: index
            });
          }})
          vegetables.forEach(function (item,index){
          options.vegetables.push({
            label: item,
            value: index
          });
        })
        grains.forEach(function (item,index){
          options.grains.push({
            label: item,
            value: index
          });
        })
        dairy.forEach(function (item,index){
          options.dairy.push({
            label: item,
            value: index
          });
        })
        fruits.forEach(function (item,index){
          options.fruits.push({
            label: item,
            value: index
          });
        })
        proteins.forEach(function (item,index){
          options.proteins.push({
            label: item,
            value: index
          });
        })
        setter(options)
    })  
}

export function getRecipe(list,setter){
  var url = 'http://40.76.48.13:80/get_recipe'
  var newlist = []

  list.forEach(function(element){

    newlist.push(element.label)
  });
  var postData = {ingredients:newlist}

  var options = {
    method: 'post',
    body: postData,
    json: true,
    url: url
  }

  request(options, function (err, res, body) {
    if (err) {
      console.error('error posting json: ', err)
      throw err
    }
    let recipe = body['recipe'].split(",")
    let idx = 0
    recipe.forEach(function(element) {
      if(idx == 0){recipe[idx] = (idx + 1) + ". " + element.charAt(0).toUpperCase() + element.slice(1)}
      else{recipe[idx] = (idx + 1) + ". " + element.charAt(2).toUpperCase() + element.slice(3)}
      idx += 1
    })
    console.log(recipe)
    setter(recipe)
  })
}