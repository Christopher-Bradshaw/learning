import * as THREE from 'three';
import { Planet } from "./solar_system_movement"

// Number of segments on our spheres
const segs = 16;

// The order of these needs to be
// a, e, i, long asc node, arg of periapse, MA0

// http://www.met.rdg.ac.uk/~ross/Astronomy/Planets.html
// const orbital_elements = {
//     "Mercury": [0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
//     "Venus": [0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
//     "Earth": [1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
//     "Mars": [1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
//     "Jupiter": [5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
//     "Saturn": [9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
//     "Uranus": [19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
//     "Neptune": [30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
//     "Pluto": [39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881],
// }

// https://ssd.jpl.nasa.gov/?bodies#elem and https://ssd.jpl.nasa.gov/txt/p_elem_t1.txt
// Note that we slightly reorder the cols
// Check our math is the same as https://ssd.jpl.nasa.gov/txt/aprx_pos_planets.pdf
const orbital_elements = {
    "Mercury": [ 0.38709927,      0.20563593,      7.00497902,     48.33076593,       77.45779628,    252.25032350],
    "Venus":   [ 0.72333566,      0.00677672,      3.39467605,     76.67984255,      131.60246718,    181.97909950],
    "Earth":   [ 1.00000261,      0.01671123,     -0.00001531,      0.0       ,      102.93768193,    100.46457166],
    "Mars":    [ 1.52371034,      0.09339410,      1.84969142,     49.55953891,      -23.94362959,     -4.55343205],
    "Jupiter": [ 5.20288700,      0.04838624,      1.30439695,    100.47390909,       14.72847983,     34.39644051],
    "Saturn":  [ 9.53667594,      0.05386179,      2.48599187,    113.66242448,       92.59887831,     49.95424423],
    "Uranus":  [19.18916464,      0.04725744,      0.77263783,     74.01692503,      170.95427630,    313.23810451],
    "Neptune": [30.06992276,      0.00859048,      1.77004347,    131.78422574,       44.96476227,    -55.12002969],
    "Pluto":   [39.48211675,      0.24882730,     17.14001206,    110.30393684,      224.06891629,    238.92903833],
}


const size = {
    "Sun": 696342,
    "Mercury": 2439.7,
    "Venus": 6051.8,
    "Earth": 6371,
    "Mars": 3389.5,
    "Jupiter": 69911,
    "Saturn": 58232, // Without rings
    "Uranus": 25362,
    "Neptune": 24622,
    "Pluto": 1188.3,
}
var km_in_au = 1.496e8
for (var k of Object.keys(size)) {
    size[k] = size[k] / km_in_au;
}

// Some things to know about textures:
// .map: The color map. This is modulated by the diffuse .color. So, if we have the color that we want in the map, set color to white (which it is by default).
// .shininess: How much specular highlight (mirror like reflection) there is. Large number - more shiny. Default is 30.

function new_planet(name, size_scale) {
    size_scale = size_scale || 1000;
    return new Planet(
        new THREE.Mesh(
            new THREE.SphereGeometry(size[name]*size_scale, segs, segs),
            new THREE.MeshPhongMaterial({
                reflectivity: 0.01,
                map: new THREE.TextureLoader().load(`../assets/${name}-small.jpg`),
                shininess: 10,
            }),
        ),
        ...orbital_elements[name],
    )
}

var sun = new THREE.Mesh(
    new THREE.SphereGeometry(size["Sun"], segs, segs),
    new THREE.MeshPhongMaterial({
        emissive: 0xffff00, // This needs to look yellow
        emissiveMap: new THREE.TextureLoader().load("../assets/Sun-small.jpg"),
        shininess: 0,
    }),
);
var sun_light = new THREE.PointLight(0xffffff, 1, 100, 2);

export { sun, sun_light, new_planet };
