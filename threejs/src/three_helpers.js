import * as THREE from "three";

function arrarr_to_vecarr(arr) {
    var vecs = [];
    for (var item of arr) {
        vecs.push(new THREE.Vector3(...item));
    }
    return vecs;
}

export { arrarr_to_vecarr };
