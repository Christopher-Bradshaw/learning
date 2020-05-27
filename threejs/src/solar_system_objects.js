import * as THREE from 'three';
import { Planet } from "./solar_system_movement"

// Number of segments on our spheres
const segs = 16;

// The order of these needs to be
// a, e, i, longitude of ascending node, longitude of periapse, mean longitude

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


// Sizes are in km. We convert to AU
var km_in_au = 1.496e8
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
for (var k of Object.keys(size)) {
    size[k] = size[k] / km_in_au;
}
const size_rings = {
    // [inner_rad, outer_rad]
    "Saturn": [70000/km_in_au, 150000/km_in_au],
}

// Some things to know about textures:
// .map: The color map. This is modulated by the diffuse .color. So, if we have the color that we want in the map, set color to white (which it is by default).
// .shininess: How much specular highlight (mirror like reflection) there is. Large number - more shiny. Default is 30.

function new_planet(name, size_scale) {
    size_scale = size_scale || 500;
    const planet_texture = new THREE.TextureLoader().load(`../assets/${name}-small.jpg`);
    planet_texture.mapping = THREE.EquirectangularReflectionMapping;
    const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(size[name]*size_scale, segs, segs),
        new THREE.MeshPhongMaterial({
            reflectivity: 0.01,
            map: planet_texture,
            shininess: 10,
        }),
    );
    // Should do this properly with the actual inclindation...
    mesh.rotation.x = Math.PI/2 + 0.2;
    const mesh_group = new THREE.Group();
    mesh_group.add(mesh);

    // This doesn't work at the moment, the texture is not applied around the ring, but across it
    if (name == "Saturn") {
        const ring_texture = new THREE.TextureLoader().load(`../assets/${name}_rings-small.jpg`);
        const ring_mesh = new THREE.Mesh(
            new THREE.RingGeometry(size_rings[name][0]*size_scale, size_rings[name][1]*size_scale, segs, segs),
            new THREE.MeshPhongMaterial({
                map: ring_texture,
                side: THREE.DoubleSide,
            }),
        );
        ring_mesh.rotation.x = 0.2;
        mesh_group.add(ring_mesh);
    }

    return new Planet(
        mesh_group,
        ...orbital_elements[name],
    )
}

function create_sun(size_scale) {
    size_scale = size_scale || 10;
    var sun = new THREE.Mesh(
        new THREE.SphereGeometry(size["Sun"] * size_scale, segs, segs),
        new THREE.MeshPhongMaterial({
            emissive: 0xffff00, // This needs to look yellow
            emissiveMap: new THREE.TextureLoader().load("../assets/Sun-small.jpg"),
            shininess: 0,
        }),
    );
    sun.rotation.x = Math.PI/2;
    var sun_light = new THREE.PointLight(0xffffff, 1, 100, 2);
    return [sun, sun_light]
}

export { create_sun, new_planet };
