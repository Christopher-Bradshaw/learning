import * as THREE from 'three';
import { Planet } from "./solar_system_movement"
import { arrarr_to_vecarr } from "./three_helpers";

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

// Further than any planets. We will stop the sun from reaching here
const bg_dist = 50;

// Sizes are in km. We convert to AU
var km_in_au = 1.496e8
const sizes = {
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
for (var k of Object.keys(sizes)) {
    sizes[k] = sizes[k] / km_in_au;
}
const size_rings = {
    // [inner_rad, outer_rad]
    "Saturn": [70000/km_in_au, 150000/km_in_au],
}

// Some things to know about textures:
// .map: The color map. This is modulated by the diffuse .color. So, if we have the color that we want in the map, set color to white (which it is by default).
// .shininess: How much specular highlight (mirror like reflection) there is. Large number - more shiny. Default is 30.

function new_trace_line(planet) {
    var line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(arrarr_to_vecarr(planet.trace_points)),
        new THREE.LineBasicMaterial({
            color: 0x0000ff,
            linewidth: 2.5,
        }),
    )
    return line;
}

var shared_sphere_geometry = new THREE.SphereBufferGeometry(1, segs, segs);

function new_planet(name, size_scale) {
    const planet_texture = new THREE.TextureLoader().load(`../assets/${name}-small.jpg`);
    planet_texture.mapping = THREE.EquirectangularReflectionMapping;
    const mesh = new THREE.Mesh(
        shared_sphere_geometry,
        new THREE.MeshPhongMaterial({
            reflectivity: 0.01,
            map: planet_texture,
            shininess: 10,
        }),
    );
    const size = sizes[name] * (size_scale || 500);
    mesh.scale.set(size, size, size);
    // Should do this properly with the actual inclindation...
    mesh.rotation.x = Math.PI/2 + 0.2;
    const mesh_group = new THREE.Group();
    mesh_group.add(mesh);

    // This doesn't work at the moment, the texture is not applied around the ring, but across it
    if (name == "Saturn") {
        const ring_texture = new THREE.TextureLoader().load(`../assets/${name}_rings-small.jpg`);
        const ring_mesh = new THREE.Mesh(
            new THREE.RingBufferGeometry(
                size_rings[name][0]*size_scale, size_rings[name][1]*size_scale, segs, segs
            ),
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
        {"show_trace": true},
    )
}

function create_sun(size_scale) {
    var mesh = new THREE.Mesh(
        shared_sphere_geometry,
        new THREE.MeshPhongMaterial({
            emissive: 0xffff00, // This needs to look yellow
            emissiveMap: new THREE.TextureLoader().load("../assets/Sun-small.jpg"),
            shininess: 0,
        }),
    );
    const size = sizes["Sun"] * (size_scale || 10);
    mesh.scale.set(size, size, size);
    mesh.rotation.x = Math.PI/2;
    var sun_light = new THREE.PointLight(0xffffff, 1, bg_dist - 1, 2);
    return [mesh, sun_light]
}

function create_background_stars() {
    var bg_texture = new THREE.TextureLoader().load("../assets/Stars2.jpg");
    bg_texture.mapping = THREE.EquirectangularReflectionMapping;
    var bg = new THREE.Mesh(
        new THREE.SphereGeometry(bg_dist, segs*4, segs*4),
        new THREE.MeshPhongMaterial({
            emissive: 0xffffff,
            emissiveMap: bg_texture,
            emissiveIntensity: 1,
            side: THREE.BackSide,
        }),
    );
    bg.rotation.x = Math.PI/2;
    return bg;
}

export { create_background_stars, create_sun, new_planet, new_trace_line };
