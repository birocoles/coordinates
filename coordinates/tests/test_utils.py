import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from functools import reduce
from .. import utils

def test_prime_radius_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, utils.prime_vertical_curv, 10, 23, latitude)


def test_prime_radius_negative_semiaxes():
    'null semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, utils.prime_vertical_curv, -10, -23, latitude)


def test_meridian_radius_bad_semiaxes():
    'major semiaxis lower than minor semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, utils.meridian_curv, 10, 23, latitude)


def test_meridian_radius_negative_semiaxes():
    'null semiaxis'
    latitude = np.ones(100)
    raises(AssertionError, utils.meridian_curv, -10, -23, latitude)


def test_relationship_curvatures():
    'verify relationship between the curvatures'
    latitude = np.linspace(-90, 90, 181)
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)
    N = utils.prime_vertical_curv(a, b, latitude)
    M = utils.meridian_curv(a, b, latitude)
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
    N_calc_0 = utils.prime_vertical_curv(a, b, 0)
    N_calc_90 = utils.prime_vertical_curv(a, b, 90)
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
    N = utils.prime_vertical_curv(a, b, lat_true)

    aae(N_true, N, decimal=15)


def test_meridian_radius_known_input():
    'verify results obtained for known input'
    a = 6378137.0
    f = 1.0/298.257223563
    b = a*(1-f)
    e2 = (a*a - b*b)/(a*a)
    M_true_0 = a*(1 - e2)
    M_true_90 = a/np.sqrt(1 - e2)
    M_calc_0 = utils.meridian_curv(a, b, 0)
    M_calc_90 = utils.meridian_curv(a, b, 90)
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
    M = utils.meridian_curv(a, b, lat_true)

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

    R = utils.rotation_NEU(latitude, longitude)

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

    R = utils.rotation_NEU(latitude, longitude)

    aae(R_true, R, decimal=15)


def test_rotation_NEU_orthogonality():
    'Rotation matrix must be mutually orthogonal'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R = utils.rotation_NEU(latitude, longitude)
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
    raises(AssertionError, utils.rotation_NEU, latitude, longitude)


def test_rotation_NED_orthogonality():
    'Rotation matrix must be mutually orthogonal'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R = utils.rotation_NED(latitude, longitude)
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
    raises(AssertionError, utils.rotation_NED, latitude, longitude)


def test_rotation_NEU_versus_NED():
    'Two first columns must be the same and the last must be opposite'
    latitude = np.array([0, -15, 22.5, -30, 45, -60, 75, -90])
    longitude = np.array([-17, 0, 30, 9, 90, 23, 180, 1])
    R_NEU = utils.rotation_NEU(latitude, longitude)
    R_NED = utils.rotation_NED(latitude, longitude)
    for R1, R2 in zip(R_NEU, R_NED):
        aae(R1[0], R2[0], decimal=15)
        aae(R1[1], R2[1], decimal=15)
        aae(R1[2], -R2[2], decimal=15)
        aae(R1[3], R2[3], decimal=15)
        aae(R1[4], R2[4], decimal=15)
        aae(R1[5], -R2[5], decimal=15)
        aae(R1[6], R2[6], decimal=15)
        aae(R1[7], -R2[7], decimal=15)


def test_unit_vector_TCS_bad_arguments():
    'inclination and declination with different number of elements'
    inclination = np.empty(12)
    declination = np.empty(10)
    raises(AssertionError, utils.unit_vector_TCS, inclination, declination)


def test_unit_vector_TCS_returns_unit_vector():
    'the computed vectos must be unitary'
    inclination, declination = np.meshgrid(np.linspace(-90, 90, 3),
                                           np.linspace(0, 180, 3))
    inclination = np.ravel(inclination)
    declination = np.ravel(declination)
    vx, vy, vz = utils.unit_vector_TCS(inclination, declination)
    norm = np.sqrt(vx*vx + vy*vy + vz*vz)
    aae(norm, np.ones_like(norm), decimal=15)


def test_rotations_Rx_Ry_Rz_orthonal():
    'Rotation matrices must be orthogonal'
    A = utils.rotation_Rx(-19)
    B = utils.rotation_Ry(34.71)
    C = utils.rotation_Rz(28)

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
    A1 = utils.rotation_Rx(-67)
    A2 = utils.rotation_Rx(67).T
    aae(A1, A2, decimal=15)

    A1 = utils.rotation_Ry(-17)
    A2 = utils.rotation_Ry(17).T
    aae(A1, A2, decimal=15)

    A1 = utils.rotation_Rz(-39)
    A2 = utils.rotation_Rz(39).T
    aae(A1, A2, decimal=15)

    A1 = utils.rotation_Rx(13)
    A2 = utils.rotation_Rx(-13).T
    aae(A1, A2, decimal=15)

    A1 = utils.rotation_Ry(8)
    A2 = utils.rotation_Ry(-8).T
    aae(A1, A2, decimal=15)

    A1 = utils.rotation_Rz(40)
    A2 = utils.rotation_Rz(-40).T
    aae(A1, A2, decimal=15)


def test_relationship_unit_vectors():
    'unit vectors triad must satisfy a mathematical relationship'

    N = 200
    latitude = np.linspace(-90, 90, N)
    longitude = np.linspace(0, 180, N)

    rX, rY, rZ = utils.unit_vector_normal(latitude, longitude)
    sX, sY, sZ = utils.unit_vector_latitude(latitude, longitude)
    wX, wY = utils.unit_vector_longitude(longitude)

    V = np.vstack([rX, rY, rZ, sX, sY, sZ, wX, wY, np.zeros(N)]).T

    for v1, v2, lat1, lat2, lon1, lon2 in zip(V[:-1], V[1:],
                                              latitude[:-1], latitude[1:],
                                              longitude[:-1], longitude[1:]):
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        V1 = np.reshape(v1, (3,3)).T
        V2 = np.reshape(v2, (3,3)).T
        R3_dlon = utils.rotation_Rz(-dlon)
        R3_dlat = utils.rotation_Rz(-dlat)
        V2_calc = reduce(np.dot, [R3_dlon, V1, R3_dlat])
        aae(V2, V2_calc, decimal=10)


def test_spherical_central_angle():
    'compare results with expected values'

    D1 = utils.spherical_central_angle(0, 0, 1, 0)
    D2 = utils.spherical_central_angle(0, 0, 0, 1)
    aae(D1, D2)

    D1 = utils.spherical_central_angle(80, 10, 81, 10)
    D2 = utils.spherical_central_angle(80, 10, 79, 10)
    aae(D1, D2)

    D1 = utils.spherical_central_angle(80, 10, 81, 10)
    D2 = utils.spherical_central_angle(80, 130.7, 79, 130.7)
    aae(D1, D2)
