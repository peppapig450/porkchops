import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from astropy.constants import G, M_sun
from astropy.coordinates import (
    solar_system_ephemeris,
    get_body_barycentric_posvel,
    CartesianRepresentation,
)
from scipy.optimize import root_scalar

# Constants
mu = G * M_sun  # Graviational parameter of the Sun


def get_position_velocity(body, time):
    with solar_system_ephemeris.set("builtin"):
        pos, vel = get_body_barycentric_posvel(body, time)

        if isinstance(pos, CartesianRepresentation) and isinstance(
            vel, CartesianRepresentation
        ):

            return pos.xyz.to(u.km).value, vel.xyz.to(u.km / u.s).value
        raise ValueError(
            "Something went wrong calcualting the barycentric position velocities for {body} and {time}"
        )


def solve_lambert(r1, r2, tof, mu):
    def kepler_eqn(x, a, b, c):
        return a * x**3 + b * x**2 + c * x - 1

    def calc_velocity_vectors(r1, r2, tof, x):
        # Calculate vector norms of r1 and r2
        norm_r1 = np.linalg.norm(r1)
        norm_r2 = np.linalg.norm(r2)

        # Calculate semi-major axis
        a = 0.5 * (norm_r1 + norm_r2 + x)

        # Calculate Lagrange coefficients
        f = 1 - x / norm_r1
        g_dot = 1 - x / norm_r2
        g = tof - np.sqrt(x**3 / mu)

        # Calculate velocity vectors
        v1 = (r2 - f * r1) / g
        v2 = (g_dot * r2 - r1) / g
        return v1, v2

    # Initial guesses
    a = (
        np.linalg.norm(r1)
        * np.linalg.norm(r2)
        * (
            1
            + np.cos(
                np.arccos(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
            )
        )
        / 2
    )
    b = -np.linalg.norm(r1) - np.linalg.norm(r2)
    c = 2 * np.sqrt(mu) * tof

    # Solve Kepler's equation
    sol = root_scalar(
        kepler_eqn,
        args=(a, b, c),
        method="brentq",
        bracket=[0, np.linalg.norm(r1) + np.linalg.norm(r2)],
    )

    if sol.converged:
        x = sol.root
        return calc_velocity_vectors(r1, r2, tof, x)
    raise ValueError("Lambert's problem did not converge.")


# Define launch and arrival dates
launch_dates = Time("2024-01-01") + np.arange(0, 365, 10) * u.day
arrival_dates = Time("2024-04-01") + np.arange(0, 365, 10) * u.day

# Create arrays to hold the delta-v values
dv = np.zeros((len(launch_dates), len(arrival_dates)))

# Calculate delta-v for each launch and arrival date combination
for i, launch_date in enumerate(launch_dates):
    r1, v1_earth = get_position_velocity("earth", launch_date)
    for j, arrival_date in enumerate(arrival_dates):
        r2, v2_mars = get_position_velocity("mars", arrival_date)
        tof = (arrival_date - launch_date).to(u.second).value

        try:
            v1, v2 = solve_lambert(r1, r2, tof, mu.value)
            dv[i, j] = np.linalg.norm(v1 - v1_earth) + np.linalg.norm(v2 - v2_mars)
        except ValueError:
            dv[i, j] = np.nan

# Create the porkchop plot
plt.figure(figsize=(10, 8))
X, Y = np.meshgrid(launch_dates.jd, arrival_dates.jd)
contour = plt.contourf(
    X, Y, dv.T, levels=np.linspace(np.nanmin(dv), np.nanmax(dv), 30), cmap="viridis"
)
plt.colorbar(contour, label="Delta-v (km/s)")
plt.xlabel("Launch Date (JD)")
plt.ylabel("Arrival Date (JD)")
plt.title("Porkchop Plot for Earth to Mars")
plt.show()
