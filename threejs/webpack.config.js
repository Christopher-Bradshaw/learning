const path = require("path");

module.exports = {
  mode: "development",
  entry: {
    scene1: "./src/scene1.js",
    scene2: "./src/scene2.js",
  },
  output: {
    filename: "[name].js",
    path: path.resolve(__dirname, "dist"),
  },
};
