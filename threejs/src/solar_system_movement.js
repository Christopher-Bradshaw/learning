import { nelderMead } from "fmin";

var { cos, sin, PI, pow } = Math;
const millis_per_year = 30 * 1000;

class Planet {
    // a: semi-major axis (in AU)
    // e: eccentricity
    // i: inclination (from the reference (ecliptic/x-y) plane)
    // long_asc (omega): longitude of ascending node. The angle from the reference direction (x axis), along the reference plane, to the ascending node
    // long_peri (w): longitude of periapse. The angle from the reference direction to periapse
    // ML0: mean longitude (from the reference position) at t=0.
    // We mostly follow https://ssd.jpl.nasa.gov/txt/aprx_pos_planets.pdf
    constructor(mesh, a, e, i, long_asc, long_peri, ML0, options) {
        this.mesh = mesh;
        this.a = a;
        this.e = e;
        this.i = this._deg_to_rad(i);
        this.long_asc = this._deg_to_rad(long_asc);
        this.long_peri = this._deg_to_rad(long_peri);
        // This is the angle between the ascending node and periapse
        this.arg_peri = (this.long_peri - this.long_asc) % (2*PI);
        // Mean anomaly (MA) is the angle from periapse.
        // To go from ML (mean longitude, angle from ref) to MA (mean anomaly, angle from peri)
        // L = long_peri + MA (ref-asc + asc-peri + peri-pos)
        this.MA0 = (this._deg_to_rad(ML0) - this.long_peri) % (2*PI);

        // t: period (in years)
        this.t = pow(this.a, 3/2);

        this.handle_options(options);

        // Set the initial position
        this.update_position(0);
        this.update_trace_points(0);
    }

    handle_options(options) {
        if (options === undefined) {
            options = {};
        }
        this.show_trace = options["show_trace"] || false;
    }

    update_trace_points(t_now) {
        const [trace_length_years, n_trace_points] = [0.2, 10];
        const trace_length_millis = trace_length_years * millis_per_year;
        const dt = trace_length_millis / n_trace_points;

        var trace_points = [];
        for (var i = 0; i < n_trace_points; i++) {
            trace_points.push(this.get_position(t_now - dt * i));
        }
        this.trace_points = trace_points;
    }

    update_position(time) {
        const [x, y, z] = this.get_position(time);
        this.update_trace_points(time);
        this.mesh.position.set(x, y, z);
    }

    get_position(time) {
        const MA = this._compute_mean_anomaly(time);
        const EA = this._compute_eccentric_anomaly(MA);
        return this._compute_xyz(EA);
    }

    _compute_mean_anomaly(time) {
        return (this.MA0 + 2*PI * time / (this.t * millis_per_year)) % (2*PI);
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

    _compute_radius(EA) {
        return this.a * (1 - this.e * cos(EA));
    }

    _compute_xyz(EA) {
        // First the positions in the plane of the orbit with the x axis towards the perihelion
        const xp = this.a * (cos(EA) - this.e);
        const yp = this.a * pow((1 - pow(this.e, 2)), 0.5) * sin(EA);
        // Now in the plane of the ecliptic, with the x axis towards the march equinox
        const x = xp * (cos(this.arg_peri) * cos(this.long_asc) - sin(this.arg_peri) * sin(this.long_asc) * cos(this.i)) -
            yp * (sin(this.arg_peri) * cos(this.long_asc) + cos(this.arg_peri) * sin(this.long_asc) * cos(this.i));
        const y = xp * (cos(this.arg_peri) * sin(this.long_asc) + sin(this.arg_peri) * cos(this.long_asc) * cos(this.i)) +
            yp * (-sin(this.arg_peri) * sin(this.long_asc) + cos(this.arg_peri) * cos(this.long_asc) * cos(this.i));
        const z = xp * (sin(this.arg_peri) * sin(this.i)) + yp * (cos(this.arg_peri) * sin(this.i))
        return [x, y, z];
    }

    _deg_to_rad(deg) {
        return (deg * PI / 180) % (2*PI);
    }

}

export { Planet, millis_per_year }
