import * as THREE from "three";
import { create_background_stars, create_sun, new_planet, new_trace_line } from "./solar_system_objects";
import { millis_per_year } from "./solar_system_movement";
import { arrarr_to_vecarr } from "./three_helpers";

THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);
trace_lines = false;

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
scene.add(sun, sun_light);

var which_planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"];
var planets = which_planets.reduce(function (res, pl) { res[pl] = new_planet(pl); return res; }, {})
for (var pl of Object.values(planets)) {
    scene.add(pl.mesh);
}
if (trace_lines) {
    var trace_lines = which_planets.reduce(function (res, pl) { res[pl] = new_trace_line(planets[pl]); return res; }, {})
    for (var tl of Object.values(trace_lines)) {
        scene.add(tl);
    }
}

var bg = create_background_stars();
scene.add(bg);

// Camera
var camera = new THREE.PerspectiveCamera(
    80, // Vertical fov in degrees
    1280/960, // Aspect ratio (width/height)
    0.01, // Near plane
    1000, // Far plane (things beyond this are not seen)
);
// camera.position.x = 2;

var animate = function (t) {
    if (paused) {
        return;
    }

    for (var pl of which_planets) {
        planets[pl].update_position(t);
        if (trace_lines) {
            trace_lines[pl].geometry.setFromPoints(arrarr_to_vecarr(planets[pl].trace_points));
            trace_lines[pl].geometry.verticesNeedUpdate = true;
        }
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
