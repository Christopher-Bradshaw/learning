const path = require("path");

module.exports = {
  mode: "development",
  entry: {
    scene1: "./src/scene1.js",
    solar_system: "./src/solar_system.js",
  },
  output: {
    filename: "[name].js",
    path: path.resolve(__dirname, "dist"),
  },
};
