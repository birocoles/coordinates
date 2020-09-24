import numpy as np
import coordinates as coord
import geodetic_elipsoids as el
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from functools import reduce


def test_prime_radius_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, coord.prime_vertical_curv, 10, 23, latitude)


def test_prime_radius_negative_semiaxes():
    'null semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, coord.prime_vertical_curv, -10, -23, latitude)


def test_meridian_radius_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, coord.meridian_curv, 10, 23, latitude)


def test_meridian_radius_negative_semiaxes():
    'null semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, coord.meridian_curv, -10, -23, latitude)


def test_relationship_curvatures():
    'verify relationship between the curvatures'
    latitude = np.linspace(-90, 90, 181)
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)
    N = coord.prime_vertical_curv(a, b, latitude)
    M = coord.meridian_curv(a, b, latitude)
    e2 = (a*a - b*b)/(a*a)
    sin2lat = np.sin(np.deg2rad(latitude))
    sin2lat *= sin2lat
    M_relationship = ((1 - e2)/(1 - e2*sin2lat))*N
    aae(M, M_relationship, decimal=8)


def test_prime_radius_known_input():
    'verify results obtained for known input'
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)
    e2 = (a*a - b*b)/(a*a)
    N_true_0 = a
    N_true_90 = a/np.sqrt(1 - e2)
    N_calc_0 = coord.prime_vertical_curv(a, b, 0)
    N_calc_90 = coord.prime_vertical_curv(a, b, 90)
    aae(N_true_0, N_calc_0, decimal=15)
    aae(N_true_90, N_calc_90, decimal=15)


def test_prime_radius_known_input_2():
    'verify results obtained for known input'
    # true geodetic coordinates
    lat_true = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # squared first eccentricity
    e2 = (a**2. - b**2.)/(a**2.)

    # true prime vertical radius of curvature
    N_true = a/np.sqrt(1 - e2*sinlat*sinlat)

    # computed prime vertical radius of curvature
    N = coord.prime_vertical_curv(a, b, lat_true)

    aae(N_true, N, decimal=15)


def test_meridian_radius_known_input():
    'verify results obtained for known input'
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)
    e2 = (a*a - b*b)/(a*a)
    M_true_0 = a*(1 - e2)
    M_true_90 = a/np.sqrt(1 - e2)
    M_calc_0 = coord.meridian_curv(a, b, 0)
    M_calc_90 = coord.meridian_curv(a, b, 90)
    aae(M_true_0, M_calc_0, decimal=15)
    aae(M_true_90, M_calc_90, decimal=8)


def test_meridian_radius_known_input_2():
    'verify results obtained for known input'
    # true geodetic coordinates
    lat_true = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # squared first eccentricity
    e2 = (a**2. - b**2.)/(a**2.)

    # true meridian radius of curvature
    M_true = (a*(1 - e2))/np.power(1 - e2*sinlat*sinlat, 1.5)

    # computed meridian radius of curvature
    M = coord.meridian_curv(a, b, lat_true)

    aae(M_true, M, decimal=8)


def test_rotation_NEU_known_values():
    'verify results obtained for known input'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([0, 0, 0, 90, 90, 90, 180, 180])

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])
    coslat = np.array([1, (np.sqrt(6) + np.sqrt(2))/4,
                      (np.sqrt(2 + np.sqrt(2)))/2, np.sqrt(3)/2,
                      np.sqrt(2)/2, 0.5, (np.sqrt(6) - np.sqrt(2))/4, 0])
    sinlon = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    coslon = np.array([1, 1, 1, 0, 0, 0, -1, -1])

    R11_true = -sinlat*coslon
    R12_true = -sinlon
    R13_true = coslat*coslon
    R21_true = -sinlat*sinlon
    R22_true = coslon
    R23_true = coslat*sinlon
    R31_true = coslat
    #R32_true = 0.
    R33_true = sinlat

    R_true = np.vstack([R11_true, R12_true, R13_true,
                        R21_true, R22_true, R23_true,
                        R31_true, R33_true]).T

    R = coord.rotation_NEU(latitude, longitude)

    aae(R_true, R, decimal=15)


def test_rotation_NEU_known_values_floats():
    'verify results obtained for known input'
    latitude = -15
    longitude = 0

    sinlat = -(np.sqrt(6) - np.sqrt(2))/4
    coslat = (np.sqrt(6) + np.sqrt(2))/4
    sinlon = 0
    coslon = 1

    R11_true = -sinlat*coslon
    R12_true = -sinlon
    R13_true = coslat*coslon
    R21_true = -sinlat*sinlon
    R22_true = coslon
    R23_true = coslat*sinlon
    R31_true = coslat
    #R32_true = 0.
    R33_true = sinlat

    R_true = np.array([[R11_true, R12_true, R13_true,
                        R21_true, R22_true, R23_true,
                        R31_true, R33_true]])

    R = coord.rotation_NEU(latitude, longitude)

    aae(R_true, R, decimal=15)


def test_rotation_NEU_orthogonality():
    'Rotation matrix must be mutually orthogonal'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R = coord.rotation_NEU(latitude, longitude)
    for Ri in R:
        M = np.array([[Ri[0], Ri[1], Ri[2]],
                      [Ri[3], Ri[4], Ri[5]],
                      [Ri[6],     0, Ri[7]]])
        aae(np.dot(M.T, M), np.identity(3), decimal=15)
        aae(np.dot(M, M.T), np.identity(3), decimal=15)


def test_rotation_NEU_lines_bad_arguments():
    'latitude and longitude with different number of elements'
    latitude = np.ones(100)
    longitude = np.zeros(34)
    raises(AssertionError, coord.rotation_NEU, latitude, longitude)


def test_rotation_NED_orthogonality():
    'Rotation matrix must be mutually orthogonal'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R = coord.rotation_NED(latitude, longitude)
    for Ri in R:
        M = np.array([[Ri[0], Ri[1], Ri[2]],
                      [Ri[3], Ri[4], Ri[5]],
                      [Ri[6],     0, Ri[7]]])
        aae(np.dot(M.T, M), np.identity(3), decimal=15)
        aae(np.dot(M, M.T), np.identity(3), decimal=15)


def test_rotation_NED_lines_bad_arguments():
    'latitude and longitude with different number of elements'
    latitude = np.ones(100)
    longitude = np.zeros(34)
    raises(AssertionError, coord.rotation_NED, latitude, longitude)


def test_rotation_NEU_versus_NED():
    'Two first columns must be the same and the last must be opposite'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R_NEU = coord.rotation_NEU(latitude, longitude)
    R_NED = coord.rotation_NED(latitude, longitude)
    for R1, R2 in zip(R_NEU, R_NED):
        aae(R1[0], R2[0], decimal=15)
        aae(R1[1], R2[1], decimal=15)
        aae(R1[2], -R2[2], decimal=15)
        aae(R1[3], R2[3], decimal=15)
        aae(R1[4], R2[4], decimal=15)
        aae(R1[5], -R2[5], decimal=15)
        aae(R1[6], R2[6], decimal=15)
        aae(R1[7], -R2[7], decimal=15)


def test_GGC2GCC_bad_coordinates():
    'heigth, latitude and longitude with different number of elements'

    major_semiaxis = 10
    minor_semiaxes = 8

    latitude = np.empty(100)
    longitude = np.empty(34)
    height = np.empty(34)
    raises(AssertionError, coord.GGC2GCC, height, latitude,
           longitude, major_semiaxis, minor_semiaxes)

    latitude = np.empty(34)
    longitude = np.empty(100)
    height = np.empty(34)
    raises(AssertionError, coord.GGC2GCC, height, latitude,
           longitude, major_semiaxis, minor_semiaxes)

    latitude = np.empty(34)
    longitude = np.empty(34)
    height = np.empty(100)
    raises(AssertionError, coord.GGC2GCC, height, latitude,
           longitude, major_semiaxis, minor_semiaxes)


def test_GGC2GCC_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'

    major_semiaxis = 17
    minor_semiaxes = 20

    latitude = np.empty(34)
    longitude = np.empty(34)
    height = np.empty(34)
    raises(AssertionError, coord.GGC2GCC, height, latitude,
           longitude, major_semiaxis, minor_semiaxes)


def test_GGC2GCC_known_input():
    'verify the computed coordinates obtained from specific input'

    # geometric height (meters), latitude (degrees) and longitude (degrees)
    h = np.array([0, 0, 0])
    lat = np.array([0, 90, 0])
    lon = np.array([0, 0, 90])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # true coordinates
    x_true = np.array([a, 0, 0])
    y_true = np.array([0, 0, a])
    z_true = np.array([0, b, 0])

    # computed coordinates
    x, y, z = coord.GGC2GCC(h, lat, lon, a, b)

    aae(x_true, x, decimal=9)
    aae(y_true, y, decimal=15)
    aae(z_true, z, decimal=15)


def test_GGC2GCC_known_input_2():
    'verify the computed coordinates obtained from specific input'

    # true geodetic coordinates
    h_true = np.array([0, -100, 200, -300, 400, -800, 1600, 7000])
    lat_true = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    lon_true = np.array([0, 0, 0, 90, 90, 90, 180, 180])

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])
    coslat = np.array([1, (np.sqrt(6) + np.sqrt(2))/4,
                      (np.sqrt(2 + np.sqrt(2)))/2, np.sqrt(3)/2,
                      np.sqrt(2)/2, 0.5, (np.sqrt(6) - np.sqrt(2))/4, 0])
    sinlon = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    coslon = np.array([1, 1, 1, 0, 0, 0, -1, -1])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # squared first eccentricity
    e2 = (a**2. - b**2.)/(a**2.)

    # prime vertical radius of curvature
    N = coord.prime_vertical_curv(a, b, lat_true)

    # true Cartesian coordinates
    x_true = (N + h_true)*coslat*coslon
    y_true = (N + h_true)*coslat*sinlon
    z_true = (N*(1 - e2) + h_true)*sinlat

    # computed Cartesian coordinates
    x, y, z = coord.GGC2GCC(h_true, lat_true, lon_true, a, b)

    aae(x_true, x, decimal=9)
    aae(y_true, y, decimal=9)
    aae(z_true, z, decimal=9)


def test_GCC2GGC_bad_coordinates():
    'x, y and z with different number of elements'

    major_semiaxis = 10
    minor_semiaxes = 8

    x = np.empty(100)
    y = np.empty(34)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC, x, y, z,
           major_semiaxis, minor_semiaxes)

    x = np.empty(34)
    y = np.empty(100)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC, x, y, z,
           major_semiaxis, minor_semiaxes)

    x = np.empty(34)
    y = np.empty(34)
    z = np.empty(100)
    raises(AssertionError, coord.GCC2GGC, x, y, z,
           major_semiaxis, minor_semiaxes)


def test_GCC2GGC_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'

    major_semiaxis = 17
    minor_semiaxes = 20

    x = np.empty(34)
    y = np.empty(34)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC, x, y, z,
           major_semiaxis, minor_semiaxes)


def test_GCC2GGC_known_input():
    'verify the computed coordinates obtained from specific input'

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # Cartesian components x, y and z (in meters)
    x = np.array([a + 700, 0, 0])
    y = np.array([0, 0, a - 1200])
    z = np.array([0, b, 0])

    # true coordinates
    h_true = np.array([700, 0, -1200])
    lat_true = np.array([0, 90, 0])
    lon_true = np.array([0, 0, 90])

    # computed coordinates
    h, lat, lon = coord.GCC2GGC(x, y, z, a, b)

    aae(h_true, h, decimal=15)
    aae(lat_true, lat, decimal=15)
    aae(lon_true, lon, decimal=15)


def test_GCC2GGC_known_input_2():
    'verify the computed coordinates obtained from specific input'

    # true geodetic coordinates
    h_true = np.array([0, -100, 200, -300, 400, -800, 1600, 7000])
    lat_true = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    lon_true = np.array([0, 0, 0, 90, 90, 90, 180, 0])

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])
    coslat = np.array([1, (np.sqrt(6) + np.sqrt(2))/4,
                      (np.sqrt(2 + np.sqrt(2)))/2, np.sqrt(3)/2,
                      np.sqrt(2)/2, 0.5, (np.sqrt(6) - np.sqrt(2))/4, 0])
    sinlon = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    coslon = np.array([1, 1, 1, 0, 0, 0, -1, -1])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # squared first eccentricity
    e2 = (a**2. - b**2.)/(a**2.)

    # prime vertical radius of curvature
    N = coord.prime_vertical_curv(a, b, lat_true)

    # true Cartesian coordinates
    x_true = (N + h_true)*coslat*coslon
    y_true = (N + h_true)*coslat*sinlon
    z_true = (N*(1 - e2) + h_true)*sinlat

    # computed geodetic coordinates
    h, lat, lon = coord.GCC2GGC(x_true, y_true, z_true, a, b)

    aae(lat_true, lat, decimal=14)
    aae(lon_true, lon, decimal=15)
    aae(h_true, h, decimal=8)


def test_GCC2GGC_approx_bad_coordinates():
    'x, y and z with different number of elements'

    major_semiaxis = 10
    minor_semiaxes = 8

    x = np.empty(100)
    y = np.empty(34)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC_approx, x, y, z,
           major_semiaxis, minor_semiaxes)

    x = np.empty(34)
    y = np.empty(100)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC_approx, x, y, z,
           major_semiaxis, minor_semiaxes)

    x = np.empty(34)
    y = np.empty(34)
    z = np.empty(100)
    raises(AssertionError, coord.GCC2GGC_approx, x, y, z,
           major_semiaxis, minor_semiaxes)


def test_GCC2GGC_approx_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'

    major_semiaxis = 17
    minor_semiaxes = 20

    x = np.empty(34)
    y = np.empty(34)
    z = np.empty(34)
    raises(AssertionError, coord.GCC2GGC_approx, x, y, z,
           major_semiaxis, minor_semiaxes)


def test_GCC2GGC_approx_known_input():
    'verify the computed coordinates obtained from specific input'

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # Cartesian components x, y and z (in meters)
    x = np.array([a + 700, 0, 0])
    y = np.array([0, 0, a - 1200])
    z = np.array([0, b, 0])

    # true coordinates
    h_true = np.array([700, 0, -1200])
    lat_true = np.array([0, 90, 0])
    lon_true = np.array([0, 0, 90])

    # computed coordinates
    h, lat, lon = coord.GCC2GGC_approx(x, y, z, a, b)

    aae(h_true, h, decimal=15)
    aae(lat_true, lat, decimal=15)
    aae(lon_true, lon, decimal=15)


def test_GCC2GGC_approx_known_input_2():
    'verify the computed coordinates obtained from specific input'

    # true geodetic coordinates
    lat_true = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    lon_true = np.array([0, 0, 0, 90, 90, 90, 180, 0])
    #h_true = np.array([0, -100, 200, -300, 400, -800, 1600, 7000])
    h_true = np.zeros_like(lat_true)

    sinlat = np.array([0, -(np.sqrt(6) - np.sqrt(2))/4,
                      (np.sqrt(2 - np.sqrt(2)))/2, -0.5,
                      np.sqrt(2)/2, -np.sqrt(3)/2,
                      (np.sqrt(6) + np.sqrt(2))/4, -1])
    coslat = np.array([1, (np.sqrt(6) + np.sqrt(2))/4,
                      (np.sqrt(2 + np.sqrt(2)))/2, np.sqrt(3)/2,
                      np.sqrt(2)/2, 0.5, (np.sqrt(6) - np.sqrt(2))/4, 0])
    sinlon = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    coslon = np.array([1, 1, 1, 0, 0, 0, -1, -1])

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # squared first eccentricity
    e2 = (a**2. - b**2.)/(a**2.)

    # prime vertical radius of curvature
    N = coord.prime_vertical_curv(a, b, lat_true)

    # true Cartesian coordinates
    x_true = (N + h_true)*coslat*coslon
    y_true = (N + h_true)*coslat*sinlon
    z_true = (N*(1 - e2) + h_true)*sinlat

    # computed geodetic coordinates
    h, lat, lon = coord.GCC2GGC_approx(x_true, y_true, z_true, a, b)

    aae(lat_true, lat, decimal=13)
    aae(lon_true, lon, decimal=15)
    aae(h_true, h, decimal=8)


def test_GCC2GGC_versus_GGC2GCC():
    'compare the results produced by GCC2GGC and GGC2GCC'

    # true geodetic coordinates
    h_true = np.linspace(-1000, 1000, 10)
    lat_true = np.linspace(-90, 89, 10)
    lon_true = np.linspace(0, 180, 10)

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # computed Cartesian coordinates
    x, y, z = coord.GGC2GCC(h_true, lat_true, lon_true, a, b)

    # computed geodetic coordinates
    h, lat, lon = coord.GCC2GGC(x, y, z, a, b)

    aae(h_true, h, decimal=8)
    aae(lat_true, lat, decimal=14)
    aae(lon_true, lon, decimal=14)


def test_GCC2GGC_versus_GCC2GGC_approx():
    'compare the results produced by GCC2GGC and \
GCC2GGC_approx'

    # true geodetic coordinates
    h_true = np.linspace(-1000, 1000, 10)
    lat_true = np.linspace(-90, 89, 10)
    lon_true = np.linspace(0, 180, 10)

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # computed Cartesian coordinates
    x, y, z = coord.GGC2GCC(h_true, lat_true, lon_true, a, b)

    # computed geodetic coordinates
    h, lat, lon = coord.GCC2GGC_approx(x, y, z, a, b)

    aae(h_true, h, decimal=8)
    aae(lat_true, lat, decimal=13)
    aae(lon_true, lon, decimal=14)


def test_GCC2TCC_bad_arguments():
    'must raise assertion error when receiving bad arguments'

    X = np.empty(34)
    Y = np.empty(34)
    Z = np.empty(34)

    X_P = 0
    Y_P = 0
    Z_P = 0
    lat_P = 0
    lon_P = 0

    raises(AssertionError, coord.GCC2TCC, np.empty(33), Y, Z, X_P, Y_P, Z_P,
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, np.empty(33), Z, X_P, Y_P, Z_P,
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, np.empty(33), X_P, Y_P, Z_P,
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, Z, np.empty(3), Y_P, Z_P,
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, Z, X_P, np.empty(3), Z_P,
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, Z, X_P, Y_P, np.empty(3),
           lat_P, lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, Z, X_P, Y_P, Z_P,
           np.empty(3), lon_P)
    raises(AssertionError, coord.GCC2TCC, X, Y, Z, X_P, Y_P, Z_P,
           lat_P, np.empty(3))


def test_GCC2TCC_known_input():
    'verify the computed coordinates obtained from specific input'

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # true Cartesian coordinates
    x_true = np.array([0, 0., 0])
    y_true = np.array([0, 0., 0.])
    z_true = np.array([0, 100, -100.])

    # geocentric Cartesian coordinates of the origin
    h_P = 100.
    lat_P = 90
    lon_P = 0

    # geodetic coordinates
    h = np.array([100, 0, 200])
    lat = np.array([90., 90, 90])
    lon = np.array([0, 0, 0])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat, lon, a, b)
    X_P, Y_P, Z_P = coord.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # computed topocentric Cartesian coordinates
    x, y, z = coord.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    aae(x_true, x, decimal=15)
    aae(y_true, y, decimal=15)
    aae(z_true, z, decimal=15)

    # true Cartesian coordinates
    x_true = np.array([0, 0., 0])
    y_true = np.array([0, 0., 0.])
    z_true = np.array([0, 100, -100.])

    # geocentric Cartesian coordinates of the origin
    h_P = 100.
    lat_P = -90
    lon_P = 0

    # geodetic coordinates
    h = np.array([100, 0, 200])
    lat = np.array([-90., -90, -90])
    lon = np.array([0, 0, 0])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat, lon, a, b)
    X_P, Y_P, Z_P = coord.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # computed topocentric Cartesian coordinates
    x, y, z = coord.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    aae(x_true, x, decimal=15)
    aae(y_true, y, decimal=15)
    aae(z_true, z, decimal=15)

    # geocentric Cartesian coordinates of the origin
    h_P = 100.
    lat_P = -56
    lon_P = 160.

    # geodetic coordinates
    h = np.array([h_P, h_P])
    lat = np.array([lat_P+0.5, lat_P-0.5])
    lon = np.array([lon_P, lon_P])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat, lon, a, b)
    X_P, Y_P, Z_P = coord.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # computed topocentric Cartesian coordinates
    x, y, z = coord.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    aae(y[0], 0, decimal=10)
    aae(y[1], 0, decimal=10)

    # geocentric Cartesian coordinates of the origin
    h_P = 100.
    lat_P = 0
    lon_P = 160.

    # geodetic coordinates
    h = np.array([h_P, h_P])
    lat = np.array([lat_P, lat_P])
    lon = np.array([lon_P+0.5, lon_P-0.5])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat, lon, a, b)
    X_P, Y_P, Z_P = coord.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # computed topocentric Cartesian coordinates
    x, y, z = coord.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    aae(x[0], 0, decimal=10)
    aae(x[1], 0, decimal=10)

    # geocentric Cartesian coordinates of the origin
    h_P = 100.
    lat_P = 40
    lon_P = -167.

    # geodetic coordinates
    h = np.array([h_P, h_P+200, h_P-300.])
    lat = np.array([lat_P, lat_P, lat_P])
    lon = np.array([lon_P, lon_P, lon_P])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat, lon, a, b)
    X_P, Y_P, Z_P = coord.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # computed topocentric Cartesian coordinates
    x, y, z = coord.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    aae(z[0], 0, decimal=15)
    aae(z[1], -200, decimal=9)
    aae(z[2], 300, decimal=9)


def test_GGC2GCC_GSC2GCC():
    'computed GCC must be the same'

    # major semiaxis, flattening, minor semiaxis and squared first eccentricity
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)

    # geodetic coordinates
    h = np.array([100, 0, -2000])
    lat_g = np.array([90., 2, -46.8])
    lon_g = np.array([0, 37, 174.6])

    # geocentric cartesian coordinates
    X, Y, Z = coord.GGC2GCC(h, lat_g, lon_g, a, b)

    # geocentric spherical coordinates
    r, lat_s, lon_s = coord.GCC2GSC(X, Y, Z)

    aae(lon_g, lon_s, decimal=12)

    lat_g2 = coord.geocentric2geodetic_latitude(lat_s, a, b)
    lat_s2 = coord.geodetic2geocentric_latitude(lat_g, a, b)
    aae(lat_g, lat_g2, decimal=4)
    aae(lat_s, lat_s2, decimal=4)

    X2, Y2, Z2 = coord.GSC2GCC(r, lat_s, lon_s)

    aae(X, X2, decimal=9)
    aae(Y, Y2, decimal=9)
    aae(Z, Z2, decimal=9)


def test_unit_vector_TCS_bad_arguments():
    'inclination and declination with different number of elements'
    inclination = np.empty(12)
    declination = np.empty(10)
    raises(AssertionError, coord.unit_vector_TCS, inclination, declination)


def test_unit_vector_TCS_returns_unit_vector():
    'the computed vectos must be unitary'
    inclination, declination = np.meshgrid(np.linspace(-90, 90, 3),
                                           np.linspace(0, 180, 3))
    inclination = np.ravel(inclination)
    declination = np.ravel(declination)
    vx, vy, vz = coord.unit_vector_TCS(inclination, declination)
    norm = np.sqrt(vx*vx + vy*vy + vz*vz)
    aae(norm, np.ones_like(norm), decimal=15)


def test_rotations_Rx_Ry_Rz_orthonal():
    'Rotation matrices must be orthogonal'
    A = coord.rotation_Rx(-19)
    B = coord.rotation_Ry(34.71)
    C = coord.rotation_Rz(28)

    aae(np.dot(A, A.T), np.dot(A.T, A), decimal=15)
    aae(np.dot(A, A.T), np.identity(3), decimal=15)
    aae(np.dot(A.T, A), np.identity(3), decimal=15)

    aae(np.dot(B, B.T), np.dot(B.T, B), decimal=15)
    aae(np.dot(B, B.T), np.identity(3), decimal=15)
    aae(np.dot(B.T, B), np.identity(3), decimal=15)

    aae(np.dot(C, C.T), np.dot(C.T, C), decimal=15)
    aae(np.dot(C, C.T), np.identity(3), decimal=15)
    aae(np.dot(C.T, C), np.identity(3), decimal=15)


def test_rotations_Rx_Ry_Rz_transposition():
    'R(-alpha) must be equal to transposed R(alpha)'
    A1 = coord.rotation_Rx(-67)
    A2 = coord.rotation_Rx(67).T
    aae(A1, A2, decimal=15)

    A1 = coord.rotation_Ry(-17)
    A2 = coord.rotation_Ry(17).T
    aae(A1, A2, decimal=15)

    A1 = coord.rotation_Rz(-39)
    A2 = coord.rotation_Rz(39).T
    aae(A1, A2, decimal=15)

    A1 = coord.rotation_Rx(13)
    A2 = coord.rotation_Rx(-13).T
    aae(A1, A2, decimal=15)

    A1 = coord.rotation_Ry(8)
    A2 = coord.rotation_Ry(-8).T
    aae(A1, A2, decimal=15)

    A1 = coord.rotation_Rz(40)
    A2 = coord.rotation_Rz(-40).T
    aae(A1, A2, decimal=15)


def test_geodetic2geocentric_latitude_known_input():
    'verify results for known input'
    latitude = np.array([-60, -45, -30, 0, 30, 45, 60])
    a, f = el.WGS84()
    b = a*(1 - f)
    aux = (b*b)/(a*a)
    sqrt3 = np.sqrt(3)
    tangent = np.array([-sqrt3, -1, -1/sqrt3, 0, 1/sqrt3, 1, sqrt3])
    true = np.rad2deg(np.arctan(aux*tangent))
    computed = coord.geodetic2geocentric_latitude(latitude, a, b)
    aae(true, computed, decimal=12)


def test_geodetic2geocentric_latitude_geocentric2geodetic_latitude():
    'verify results for known input'
    geodetic_latitude = np.array([-60, -45, -30, 0, 30, 45, 60])
    a, f = el.WGS84()
    b = a*(1 - f)
    geocentric_latitude = coord.geodetic2geocentric_latitude(geodetic_latitude,
                                                             a, b)
    geodetic_latitude2 = coord.geocentric2geodetic_latitude(geocentric_latitude,
                                                            a, b)
    aae(geodetic_latitude, geodetic_latitude2, decimal=12)


def test_geodetic2geocentric_latitude_angular_diff():
    'verify the angular difference between radial and normal unit vectors'

    a, f = el.WGS84()
    b = a*(1 - f)

    geodetic_latitude = 45
    longitude = 0

    # normal unit vector
    u = coord.unit_vector_normal(geodetic_latitude, longitude)

    geocentric_latitude = coord.geodetic2geocentric_latitude(geodetic_latitude,
                                                             a, b)

    diff1 = geocentric_latitude - geodetic_latitude

    # radial unit vector
    geocentric_colatitude = 90 - geocentric_latitude
    cos_colat = np.cos(np.deg2rad(geocentric_colatitude))
    sin_colat = np.sin(np.deg2rad(geocentric_colatitude))
    cos_lon = np.cos(np.deg2rad(longitude))
    sin_lon = np.sin(np.deg2rad(longitude))
    r = np.array([sin_colat*cos_lon , sin_colat*sin_lon, cos_colat])

    diff2 = np.rad2deg(np.arccos(np.dot(u, r)))

    aae(diff1, -np.sign(geocentric_latitude)*diff2, decimal=11)

    geodetic_latitude = -15
    longitude = 67

    # normal unit vector
    u = coord.unit_vector_normal(geodetic_latitude, longitude)

    geocentric_latitude = coord.geodetic2geocentric_latitude(geodetic_latitude,
                                                             a, b)

    diff1 = geocentric_latitude - geodetic_latitude

    # radial unit vector
    geocentric_colatitude = 90 - geocentric_latitude
    cos_colat = np.cos(np.deg2rad(geocentric_colatitude))
    sin_colat = np.sin(np.deg2rad(geocentric_colatitude))
    cos_lon = np.cos(np.deg2rad(longitude))
    sin_lon = np.sin(np.deg2rad(longitude))
    r = np.array([sin_colat*cos_lon , sin_colat*sin_lon, cos_colat])

    diff2 = np.rad2deg(np.arccos(np.dot(u, r)))

    aae(diff1, -np.sign(geocentric_latitude)*diff2, decimal=11)

    geodetic_latitude = 70
    longitude = -18

    # normal unit vector
    u = coord.unit_vector_normal(geodetic_latitude, longitude)

    geocentric_latitude = coord.geodetic2geocentric_latitude(geodetic_latitude,
                                                             a, b)

    diff1 = geocentric_latitude - geodetic_latitude

    # radial unit vector
    geocentric_colatitude = 90 - geocentric_latitude
    cos_colat = np.cos(np.deg2rad(geocentric_colatitude))
    sin_colat = np.sin(np.deg2rad(geocentric_colatitude))
    cos_lon = np.cos(np.deg2rad(longitude))
    sin_lon = np.sin(np.deg2rad(longitude))
    r = np.array([sin_colat*cos_lon , sin_colat*sin_lon, cos_colat])

    diff2 = np.rad2deg(np.arccos(np.dot(u, r)))

    aae(diff1, -np.sign(geocentric_latitude)*diff2, decimal=11)


def test_GSC2GCC_known_values():
    'computed values must be equal to reference values'

    tol = 13

    radius = 100
    latitude = 90
    longitude = 0
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, 0, decimal=tol)
    aae(Y, 0, decimal=tol)
    aae(Z, radius, decimal=tol)

    radius = 100
    latitude = -90
    longitude = 0
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, 0, decimal=tol)
    aae(Y, 0, decimal=tol)
    aae(Z, -radius, decimal=tol)

    radius = 100
    latitude = 0
    longitude = 0
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, radius, decimal=tol)
    aae(Y, 0, decimal=tol)
    aae(Z, 0, decimal=tol)

    radius = 100
    latitude = 0
    longitude = 180
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, -radius, decimal=tol)
    aae(Y, 0, decimal=tol)
    aae(Z, 0, decimal=tol)

    radius = 100
    latitude = 0
    longitude = 90
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, 0, decimal=tol)
    aae(Y, radius, decimal=tol)
    aae(Z, 0, decimal=tol)

    radius = 100
    latitude = 0
    longitude = -90
    X, Y, Z = coord.GSC2GCC(radius, latitude, longitude)
    aae(X, 0, decimal=tol)
    aae(Y, -radius, decimal=tol)
    aae(Z, 0, decimal=tol)


def test_GSC2GCC_GCC2GSC():
    'compare the transformed coordinates'

    N = 16

    radius = np.zeros(N) + 6371000
    latitude = np.linspace(-90, 90, N)
    longitude = np.linspace(-34, 70, N)
    X1, Y1, Z1 = coord.GSC2GCC(radius, latitude, longitude)
    radius1, latitude1, longitude1 = coord.GCC2GSC(X1, Y1, Z1)
    aae(radius, radius1, decimal=9)
    aae(latitude, latitude1, decimal=10)
    aae(longitude, longitude1, decimal=10)


def test_relationship_unit_vectors():
    'unit vectors triad must satisfy a mathematical relationship'

    N = 200
    latitude = np.linspace(-90, 90, N)
    longitude = np.linspace(0, 180, N)

    rX, rY, rZ = coord.unit_vector_normal(latitude, longitude)
    sX, sY, sZ = coord.unit_vector_latitude(latitude, longitude)
    wX, wY = coord.unit_vector_longitude(longitude)

    V = np.vstack([rX, rY, rZ, sX, sY, sZ, wX, wY, np.zeros(N)]).T

    for v1, v2, lat1, lat2, lon1, lon2 in zip(V[:-1], V[1:],
                                              latitude[:-1], latitude[1:],
                                              longitude[:-1], longitude[1:]):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        V1 = np.reshape(v1, (3,3)).T
        V2 = np.reshape(v2, (3,3)).T
        R3_dlon = coord.rotation_Rz(-dlon)
        R3_dlat = coord.rotation_Rz(-dlat)
        V2_calc = reduce(np.dot, [R3_dlon, V1, R3_dlat])
        aae(V2, V2_calc, decimal=10)


def test_spherical_central_angle():
    'compare results with expected values'

    D1 = coord.spherical_central_angle(0, 0, 1, 0)
    D2 = coord.spherical_central_angle(0, 0, 0, 1)
    aae(D1, D2)

    D1 = coord.spherical_central_angle(80, 10, 81, 10)
    D2 = coord.spherical_central_angle(80, 10, 79, 10)
    aae(D1, D2)

    D1 = coord.spherical_central_angle(80, 10, 81, 10)
    D2 = coord.spherical_central_angle(80, 130.7, 79, 130.7)
    aae(D1, D2)


def test_molodensky_reference_value():
    'compare results with https://www.ufrgs.br/lageo/calculos/refgeo_g.php'
    # WGS84 to SAD69
    a1, f1 = el.WGS84()
    b1 = a1*(1 - f1)
    a2, f2 = el.SAD69()
    b2 = a2*(1 - f2)
    dX, dY, dZ = el.WGS84_SAD69()

    # WGS84 coordinates
    h1 = np.array([180, 0])
    lat1 = np.array([-27, 40])
    lon1 = np.array([-50, 35])

    # SAD69 coordinates
    h2 = np.array([180.916, 42.0473])
    lat2 = np.array([-26.999504764, 39.999968436])
    lon2 = np.array([-49.9995122, 34.999508925])

    h, lat, lon = coord.molodensky_complete(
        h1, lat1, lon1, a1, f1, a2, f2, dX, dY, dZ
    )
    aae(h2, h, decimal=3)
    aae(lat2, lat, decimal=3)
    aae(lon2, lon, decimal=3)
