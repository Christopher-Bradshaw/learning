import * as THREE from 'three';

// This is what we will use to render the scene. It uses WebGL
var renderer = new THREE.WebGLRenderer();
// Set the size (in pixels) of the renderer canvas
renderer.setSize(window.innerWidth, window.innerHeight);
// And we just add this to the end of the html body
document.body.appendChild(renderer.domElement);


// Now we construct the scene
const scene = new THREE.Scene();

var geometry = new THREE.BoxGeometry();
var material = new THREE.MeshBasicMaterial( { color: 0xf00000 } );
var cube = new THREE.Mesh(geometry, material );
scene.add(cube);

// And the camera with which we will view the scene
// Note that this inherits from Camera and from Object3D
var camera = new THREE.PerspectiveCamera(
    75, // Vertical fov in degrees
    window.innerWidth/window.innerHeight, // Aspect ratio (width/height)
    0.1, // Near plane
    1000, // Far plane (things beyond this are not seen)
);
// The position is something that can be set on all Object3D things
// Positive X is to the right
// Positive Y is up
// Positive Z is out of the screen
camera.position.z = 5;

var animate = function (t) {
    // This is a browser function
    // https://developer.mozilla.org/en-US/docs/Web/API/window/requestAnimationFrame
    // Tells the browser to call this function before the next repaint
    // N.B. that this is an async function. I'm not 100% clear on this, but we don't
    // immediately re-enter animate (else we would overflow the call stack and never do
    // anythin). Instead, the synchronous code in animate executes, and when the engine
    // next pulls something off the event look, this is waiting there.
    // This can therefore be put at the start or the end of the animate function. It makes
    // absolutely no difference.
    requestAnimationFrame(animate);
    // The only argument to this function is a time stamp. A reason to use
    // requestAnimationFrame (rather than manually just time.sleep) is that this
    // intelligently stops/slows down animations when you tab away
    console.log(t);

    // We can now change things about our scene/camera.
    cube.rotation.x += 0.01;
    cube.rotation.y += 0.01;

    camera.position.x += 0.01;
    camera.position.y += 0.01;

    // Finally we actually render the scene, given the camera
    renderer.render(scene, camera);
};

animate();
