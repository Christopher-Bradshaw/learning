import * as THREE from "three";
import { create_sun, new_planet } from "./solar_system_objects";
import { millis_per_year } from "./solar_system_movement";

THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);

// Renderer
var canvas = document.getElementById("threeCanvas");
var renderer = new THREE.WebGLRenderer({
    antialias: true,
    canvas,
});

renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
var axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Size units are in AU
const [sun, sun_light] = create_sun();
sun.position.set(0, 0, 0);
sun_light.position.set(0, 0, 0);
var planets = {
    "Mercury": new_planet("Mercury"),
    "Venus": new_planet("Venus"),
    "Earth": new_planet("Earth"),
    "Mars": new_planet("Mars"),
    "Jupiter": new_planet("Jupiter"),
    "Saturn": new_planet("Saturn"),
    "Uranus": new_planet("Uranus"),
    "Neptune": new_planet("Neptune"),
};
scene.add(sun, sun_light);
for (var pl of Object.values(planets)) {
    console.log(pl);
    scene.add(pl.mesh);
}

// Camera
var camera = new THREE.PerspectiveCamera(
    60, // Vertical fov in degrees
    2, // Aspect ratio (width/height)
    0.01, // Near plane
    1000, // Far plane (things beyond this are not seen)
);
// camera.position.z = 15;
// camera.position.x = 4;

var animate = function (t) {
    if (paused) {
        return;
    }

    for (var pl of Object.values(planets)) {
        pl.update_position(t);
    }
    camera.position.set(
        ...Object.values(planets["Earth"].mesh.position)
    );
    camera.lookAt(sun.position);

    // Set the text
    textElem.innerHTML = (2000 + t/millis_per_year).toFixed(2);

    // Set debug
    if (iter % n_prev == 0) {
        debugElem.innerHTML = `FPS: ${(1000 * n_prev / (t - t_prev)).toFixed(0)}`;
        t_prev = t;
    }
    iter += 1;

    // Finally we actually render the scene, given the camera
    renderer.render(scene, camera);

    // And animate
    requestAnimationFrame(animate);
}


// Extra things
var textElem = document.getElementById("threeText");
var debugElem = document.getElementById("threeDebug");
var [t_prev, n_prev, iter] = [0, 10, 0];
var paused = false;
var pauseElem = document.getElementById("threePauseButton");
pauseElem.onclick = function() {
    paused = !paused;
    requestAnimationFrame(animate);
}

// Animate
requestAnimationFrame(animate);
