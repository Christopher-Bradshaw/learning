import { nelderMead } from "fmin";

var { cos, sin, tan, atan, PI, pow } = Math;
const millis_per_year = 10 * 1000;

class Planet {
    // a: semi-major axis (in AU)
    // e: eccentricity
    // i: inclination (from the reference (ecliptic/x-y) plane)
    // omega: longitude of ascending node. The angle from the reference direction (x axis), along the reference plane, to the ascending node
    // w: argument of periapse. The angle from the ascending node (in the plane of the orbit) to periapse
    // MA0: mean anomaly at t=0. The angle between periapse and the current position. Not a true angle
    constructor(mesh, a, e, i, omega, w, MA0) {
        this.mesh = mesh;
        this.a = a;
        this.e = e;
        this.i = this._deg_to_rad(i);
        this.omega = this._deg_to_rad(omega);
        this.w = this._deg_to_rad(w);
        this.MA0 = this._deg_to_rad(MA0);
        // t: period (in years)
        this.t = pow(this.a, 3/2);

        // TMP
        this.t_last_print = 0;

        // Set the initial position
        this.update_position(0);
    }


    update_position(time) {
        const MA = this._compute_mean_anomaly(time);
        const EA = this._compute_eccentric_anomaly(MA);
        const TA = this._compute_true_anomaly(EA);
        const r = this._compute_radius(EA);
        const [x, y, z] = this._compute_xyz(r, TA);

        if (time - this.t_last_print > 1000) {
            // console.log(time, MA, EA, r, x, y, z);
            this.t_last_print = time;
        }
        this.mesh.position.set(x, y, z);
    }

    _compute_mean_anomaly(time) {
        // Starting from the reference angle, the position at t0 is the sum of,
        // ref to asc (omega), asc to per (w), peri to position (MA0)
        const ang0 = this.MA0 - this.w - this.omega;
        return (ang0 + 2*PI * time / (this.t * millis_per_year)) % (2*PI);
    }

    // The EA is related to the MA by,
    // MA = EA - e*sin(EA), where e is the eccentricity, not Euler's e.
    _compute_eccentric_anomaly(MA) {
        // Use an arrow function to keep `this` global
        var f = (EA_guess) => pow(
            EA_guess[0] - this.e * sin(EA_guess[0]) - MA,
            2,
        );
        return nelderMead(f, [MA]).x[0];
    }

    _compute_true_anomaly(EA) {
        return 2 * atan(pow((1 + this.e) / (1 - this.e), 0.5) * tan(EA/2));
    }

    _compute_radius(EA) {
        return this.a * (1 - this.e * cos(EA));
    }

    _compute_xyz(r, TA) {
        var x = r * (
            cos(this.omega) * cos(this.w + TA) -
            sin(this.omega) * sin(this.w + TA)*cos(this.i)
        )
        var y = r * (
            sin(this.omega) * cos(this.w + TA) +
            cos(this.omega) * sin(this.w + TA)*cos(this.i)
        )
        var z = r * (
            sin(this.i) * sin(this.w + TA)
        )

        return [x, y, z];
    }

    _deg_to_rad(deg) {
        return deg * PI / 180;
    }

}

export { Planet, millis_per_year }
