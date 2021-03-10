# ThreeJS

## Hellow world

### Setup

* Install three, `npm install three`
* Install webpack (possibly just do this globally), `npm install webpack webpack-cli`

### Hello world code

* Create `src/scene1.js`. Populate it with the basic [three.js example](https://threejs.org/docs/#manual/en/introduction/Creating-a-scene)
* Run `./node_modules/.bin/webpack src/scene1.js --output-filename scene1.js`. This puts the packed javascript file in `dist/scene.js`
    * You can add a `--watch` to this to auto update!
* Create the really simple [base html file](index.html). This literally just includes our packed javascript file.

### Bit more webpack config

* Then go create the [webpack config](./webpack.config.js). You can now just run `webpack --watch`.

### Local server

I just edit `index.html` to include the right `./dist/xxx.js` file. Then spin up a local server with python.

## Little useful things

* positive x is to the right, positive y is up, positive z is out of the screen

## Resources

https://threejsfundamentals.org/threejs/lessons/threejs-fundamentals.html
