import * as THREE from 'three';


// Renderer
var renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
var axesHelper = new THREE.AxesHelper(5);
scene.add(axesHelper);

// Size units are in AU

const r_sun = 0.1;
var sun = new THREE.Mesh(
    new THREE.SphereGeometry(r_sun, 16, 16),
    new THREE.MeshBasicMaterial({color: 0xffff00}),
);

const r_earth = 0.05;
var earth = new THREE.Mesh(
    new THREE.SphereGeometry(r_earth, 16, 16),
    new THREE.MeshBasicMaterial({color: 0x00ffff}),
);
earth.position.x = 1;

var earth_orbit = new THREE.Mesh(
    new THREE.TorusGeometry(1, 0.01, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0x00ffff } ),
);

scene.add(sun, earth, earth_orbit);

// Camera
var camera = new THREE.PerspectiveCamera(
    75, // Vertical fov in degrees
    window.innerWidth/window.innerHeight, // Aspect ratio (width/height)
    0.1, // Near plane
    1000, // Far plane (things beyond this are not seen)
);
camera.position.z = 5;
camera.position.y = 2;


var animate = function () {
    requestAnimationFrame(animate);

    var camera_rot = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3( 0, 1, 0 ), // Axis to rotate around
        0.005, // The amount of rotation in radians
    );

    // Rotate camera
    camera.position.applyQuaternion(camera_rot);
    camera.lookAt(scene.position);

    // Finally we actually render the scene, given the camera
    renderer.render(scene, camera);
}

console.log(earth_orbit.quaternion);
console.log(earth_orbit.getWorldDirection());
console.log(earth_orbit.worldDirection);

var quaternion = new THREE.Quaternion();
console.log(quaternion);
earth_orbit.setRotationFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
animate();
