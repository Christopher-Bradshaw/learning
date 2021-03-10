import * as THREE from "three";
import { sun, mercury } from "./objects";

// Renderer
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
var axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Size units are in AU
sun.position.set(0, 0, 0);
var sun_light = new THREE.PointLight(0xffffff, 1, 100, 2);

mercury.position.set(0, 0, 0.3);

const r_earth = 0.05;
var earth = new THREE.Mesh(
    new THREE.SphereGeometry(r_earth, 16, 16),
    new THREE.MeshBasicMaterial({color: 0x00ffff}),
);
earth.position.x = 1;

var earth_orbit = new THREE.Mesh(
    new THREE.TorusGeometry(1, 0.01, 128, 128),
    new THREE.MeshBasicMaterial({ color: 0x00ffff } ),
);

scene.add(sun, sun_light, mercury);

// Camera
var camera = new THREE.PerspectiveCamera(
    90, // Vertical fov in degrees
    2, // Aspect ratio (width/height)
    0.00001, // Near plane
    1000, // Far plane (things beyond this are not seen)
);
camera.position.z = 5;
camera.position.y = 2;

var animate = function () {
    requestAnimationFrame(animate);

    var earth_rot = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3( 0, 1, 0 ), // Axis to rotate around
        -0.005, // The amount of rotation in radians
    );

    // Rotate earth and camera
    earth.position.applyQuaternion(earth_rot);
    camera.position.set(earth.position.x, earth.position.y+r_earth, earth.position.z);
    camera.lookAt(sun.position);

    // Finally we actually render the scene, given the camera
    renderer.render(scene, camera);
}

earth_orbit.setRotationFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
animate();
