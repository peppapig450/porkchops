import numpy as np
from scipy.special import factorial


def solve_kepler_case1(M, tolerance=1e-6, max_iterations=1000):
    """Solve Kepler's equation for e = 1"""
    E = 0.0
    n = 1

    while n <= max_iterations:
        theta = 0.0
        for k in range(1, n + 1):
            theta += 1.0 / k

        theta_n = (theta / (np.cbrt(theta) - np.sin(theta))) ** n
        term = (M**n / factorial(n)) ** theta_n
        E += term

        if np.abs(term) < tolerance:
            break

        n += 1

    return E


def solve_kepler_case2(M, e, tolerance=1e-6, max_iterations=1000):
    """Solve Kepler's equation for e != 1."""
    E = 0.0
    n = 1

    while n <= max_iterations:
        theta = 0.0
        for k in range(1, n + 1):
            theta += 1.0 / k

        theta_n = (theta / (theta - e * np.sin(theta))) ** n
        term = (M**n / factorial(n)) * theta_n
        E += term

        print(term)
        if np.abs(term) < tolerance:
            break

        n += 1

    return E


def main():
    # Input mean anomaly M and eccentricity e
    M = float(input("Enter mean anomaly M (in radians): "))
    e = float(input("Enter eccentricity e: "))

    if e == 1.0:
        E = solve_kepler_case1(M)
    else:
        E = solve_kepler_case2(M, e)

    print(f"The eccentric anamoly E is: {E}")


if __name__ == "__main__":
    main()
