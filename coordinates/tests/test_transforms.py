import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
from .. import transforms as tr


#### auxiliary functions


def test_elipsoide_entrada_invalida():
    "datum inválido"

    # string inválida
    datum_invalida = '1234'
    raises(ValueError, tr.elipsoid, datum_invalida)

    # inteiro
    datum_invalida = 12
    raises(ValueError, tr.elipsoid, datum_invalida)

    # tupla
    datum_invalida = ('WGS84',)
    raises(ValueError, tr.elipsoid, datum_invalida)

    # lista
    datum_invalida = ["GRS80",]
    raises(ValueError, tr.elipsoid, datum_invalida)


def test_raio_curvatura_primeiro_vertical_entrada_invalida():
    "latitude que não pode ser convertida para array 1D de floats"

    datum_valido = "SAD69"

    # string
    latitude_invalida = '1234'
    raises(
        ValueError, 
        tr.prime_vertical_curv, 
        latitude_invalida,
        datum_valido
        )

    # lista com vários tipos
    latitude_invalida = np.array(['N45', 45, 45.0])
    raises(
        ValueError, 
        tr.prime_vertical_curv, 
        latitude_invalida,
        datum_valido
        )

    # matriz de floats
    latitude_invalida = np.ones((3,2))
    raises(
        ValueError, 
        tr.prime_vertical_curv, 
        latitude_invalida,
        datum_valido
        )


#### coordinate transformations


def test_GG2GC_entrada_invalida():
    'pontos que não podem ser convertidos para numpy array 2D de floats '

    datum_valido = "SAD69"

    # string
    pontos_invalidos = '1234'
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )


def test_GG2GC_pontos_conhecidos():
    'verifica coordenadas calculadas para entrada conhecida'

    # datum
    datum = "WGS84"

    # altitude, latitude e longitude
    pontos = np.array([
        [0, 0, 0], [0, 90, 0], [100.0, 0, 90]
        ])

    # Extrai os parâmetros do datum
    a, f = tr.elipsoid(datum=datum)
    b = a * (1. - f) # semi-eixo menor

    # coordenadas verdadeiras
    coordenadas_verdadeiras = np.array([
        [a, 0., 0.], [0., 0., b], [0., a+100.0, 0.]
        ])

    # coordenadas calculadas
    coordenadas = tr.GG2GC(pontos, datum)

    aae(coordenadas_verdadeiras, coordenadas, decimal=6)


def test_GG2GC_invalid_points():
    'pontos que não podem ser convertidos para numpy array 2D de floats'

    datum_valido = "SAD69"

    # string
    pontos_invalidos = '1234'
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(
        ValueError, 
        tr.GG2GC, 
        pontos_invalidos,
        datum_valido
        )


def test_GG2GC_invalid_latitude_longitude():
    'invalud latitude and longitude values'

    # array with latitude values not in the interval [-90°, 90°]
    invalid_points = np.array([
        [10.0, -90.0, 12.0],
        [10.0, -91.0, 12.0],
        [10.0,  90.0, 12.0],
        ])
    raises(ValueError, tr.GG2GC, invalid_points, "GRS80")

    # array with longitude values not in the interval [-180°, 180°]
    invalid_points = np.array([
        [10.0,  30.0, 120.0],
        [10.0, -80.0,   0.0],
        [10.0,  90.0, 220.0],
        ])
    raises(ValueError, tr.GG2GC, invalid_points, "WGS84")


def test_GC2GG_itmax_invalido():
    'itmax diferente de inteiro positivo'

    datum_valido = "SAD69"
    pontos_validos = np.ones((5,3))

    # string
    itmax_invalido = '1234'
    raises(
        ValueError, 
        tr.GC2GG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # list
    itmax_invalido = [3,]
    raises(
        ValueError, 
        tr.GC2GG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # inteiro negativo
    itmax_invalido = -3
    raises(
        ValueError, 
        tr.GC2GG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # inteiro nulo
    itmax_invalido = 0
    raises(
        ValueError, 
        tr.GC2GG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )


def test_GG2GC_GC2GG():
    "compara coordenadas calculadas com verdadeiras"

    # coordenadas geodesicas verdadeiras
    coordenadas_verdadeiras = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0,    0,   90,  90,   90,  180,    0]
        ]).T

    # datum 
    datum = "WGS84"

    coordenadas_Cartesianas_calculadas = tr.GG2GC(
        points=coordenadas_verdadeiras, 
        datum=datum
        )

    coordenadas_geodesicas_calculadas = tr.GC2GG(
        points=coordenadas_Cartesianas_calculadas,
        datum=datum,
        itmax=5)

    aae(coordenadas_verdadeiras, coordenadas_geodesicas_calculadas, decimal=6)


def test_GS2GC_invalid_input():
    'points cannot be converted to a 2D numpy array 2D of floats'

    # string
    invalid_points = '1234'
    raises(ValueError, tr.GS2GC, invalid_points)

    # list
    invalid_points = [34., -1, 0.4]
    raises(ValueError, tr.GS2GC, invalid_points)

    # array with number of columns different from 3
    invalid_points = np.ones((3,2))
    raises(ValueError, tr.GS2GC, invalid_points)

    # array with number of columns different from 3
    invalid_points = np.ones((3,2))
    raises(ValueError, tr.GS2GC, invalid_points)


def test_GS2GC_invalid_latitude_longitude():
    'points with invalid latitude and longitude values'

    # array with latitude values not in the interval [-90°, 90°]
    invalid_points = np.array([
        [10000.0, -90.0, 12.0],
        [10000.0, -91.0, 12.0],
        [10000.0,  90.0, 12.0],
        ])
    raises(ValueError, tr.GS2GC, invalid_points)

    # array with longitude values not in the interval [-180°, 180°]
    invalid_points = np.array([
        [10000.0,  30.0, 120.0],
        [10000.0, -80.0,   0.0],
        [10000.0,  90.0, 220.0],
        ])
    raises(ValueError, tr.GS2GC, invalid_points)


def test_GS2GC_known_points():
    'verify computed coordinates for known input'

    # r, latitude and longitude
    points = np.array([
        [6371000., 0., 0.], 
        [6371000., 90., 0.], 
        [6371100., 0., 90.]
        ])

    # true coordinates X, Y and Z
    true = np.array([
        [6371000., 0., 0.], 
        [0., 0., 6371000.], 
        [0., 6371100.0, 0.]
        ])

    # computed coordinates X, Y and Z
    computed = tr.GS2GC(points)

    aae(computed, true, decimal=6)


def test_GC2GS_invalid_input():
    'points cannot be converted to a 2D numpy array 2D of floats'

    # string
    invalid_points = '1234'
    raises(ValueError, tr.GC2GS, invalid_points)

    # list
    invalid_points = [34., -1, 0.4]
    raises(ValueError, tr.GC2GS, invalid_points)

    # array with number of columns different from 3
    invalid_points = np.ones((3,2))
    raises(ValueError, tr.GC2GS, invalid_points)


def test_GC2GS_known_points():
    'verify computed coordinates for known input'

    # X, Y and Z
    points = np.array([
        [6371000., 0., 0.], 
        [0., 0., 6371000.], 
        [0., 6371100., 0.]
        ])

    # true coordinates r, latitude and longitude
    true = np.array([
        [6371000., 0., 0.], 
        [6371000., 90., 0.], 
        [6371100., 0., 90.]
        ])


    # computed coordinates r, latitude and longitude
    computed = tr.GC2GS(points)

    aae(computed, true, decimal=6)


def test_GS2GC_GC2GS():
    "compare computed and true GS coordinates"

    # true GS coordinates
    true_GS_coordinates = np.array([
        [0, -100,  200,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0, -135,  135,   90, -90,   90,  180,    0]
        ]).T
    true_GS_coordinates[:,0] += 6371000.0

    computed_GC_coordinates = tr.GS2GC(points=true_GS_coordinates)

    computed_GS_coordinates = tr.GC2GS(points=computed_GC_coordinates)

    aae(computed_GS_coordinates, true_GS_coordinates, decimal=4)


def test_GC2TC_invalid_origin():
    'points cannot be converted to a 2D numpy array 2D of floats'

    valid_points = np.ones((4,3))

    # string
    invalid_origin = '1234'
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "WGS84")

    # array with number of columns different from 3
    invalid_origin = np.ones((3,2))
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "SAD69")


def test_GC2TC_invalid_origin_latlon():
    'invalid latitude and longitude values of the origin'

    valid_points = np.ones((4,3))

    # invalid latitude
    invalid_origin = (-10., -91.0, 32.0)
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "WGS84")

    # invalid latitude
    invalid_origin = [-10., 122.0, 32.0]
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "GRS80")

    # invalid longitude
    invalid_origin = (-10., -11.0, 232.0)
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "WGS84")

    # invalid longitude
    invalid_origin = [-10., 22.0, -182.0]
    raises(ValueError, tr.GC2TC, valid_points, invalid_origin, "SAD69")


def test_GC2TC_known_input():
    'verify the computed coordinates obtained from specific input'

    # geodetic datum
    valid_datum = "WGS84"

    # true TC coordinates
    true_TC_coordinates = np.array([
        [0.0, 0.0,    0.0],
        [0.0, 0.0,  100.0],
        [0.0, 0.0, -100.0]
        ])

    # origin
    origin = (0.0, 0.0, 90.0)

    # true GG coordinates
    true_GG_coordinates = np.array([
        [   0.0, 0.0, 90.0],
        [-100.0, 0.0, 90.0],
        [ 100.0, 0.0, 90.0]
        ])

    # computed GC coordinates
    computed_GC_coordinates = tr.GG2GC(true_GG_coordinates, valid_datum)

    # computed TC coordinates
    computed_TC_coordinates = tr.GC2TC(computed_GC_coordinates, origin, valid_datum)

    aae(computed_TC_coordinates, true_TC_coordinates, decimal=15)

    # # true Cartesian coordinates
    # x_true = np.array([0, 0., 0])
    # y_true = np.array([0, 0., 0.])
    # z_true = np.array([0, 100, -100.])

    # # geocentric Cartesian coordinates of the origin
    # h_P = 100.
    # lat_P = -90
    # lon_P = 0

    # # geodetic coordinates
    # h = np.array([100, 0, 200])
    # lat = np.array([-90., -90, -90])
    # lon = np.array([0, 0, 0])

    # # geocentric cartesian coordinates
    # X, Y, Z = tr.GGC2GCC(h, lat, lon, a, b)
    # X_P, Y_P, Z_P = tr.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # # computed topocentric Cartesian coordinates
    # x, y, z = tr.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    # aae(x_true, x, decimal=15)
    # aae(y_true, y, decimal=15)
    # aae(z_true, z, decimal=15)

    # # geocentric Cartesian coordinates of the origin
    # h_P = 100.
    # lat_P = -56
    # lon_P = 160.

    # # geodetic coordinates
    # h = np.array([h_P, h_P])
    # lat = np.array([lat_P+0.5, lat_P-0.5])
    # lon = np.array([lon_P, lon_P])

    # # geocentric cartesian coordinates
    # X, Y, Z = tr.GGC2GCC(h, lat, lon, a, b)
    # X_P, Y_P, Z_P = tr.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # # computed topocentric Cartesian coordinates
    # x, y, z = tr.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    # aae(y[0], 0, decimal=10)
    # aae(y[1], 0, decimal=10)

    # # geocentric Cartesian coordinates of the origin
    # h_P = 100.
    # lat_P = 0
    # lon_P = 160.

    # # geodetic coordinates
    # h = np.array([h_P, h_P])
    # lat = np.array([lat_P, lat_P])
    # lon = np.array([lon_P+0.5, lon_P-0.5])

    # # geocentric cartesian coordinates
    # X, Y, Z = tr.GGC2GCC(h, lat, lon, a, b)
    # X_P, Y_P, Z_P = tr.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # # computed topocentric Cartesian coordinates
    # x, y, z = tr.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    # aae(x[0], 0, decimal=10)
    # aae(x[1], 0, decimal=10)

    # # geocentric Cartesian coordinates of the origin
    # h_P = 100.
    # lat_P = 40
    # lon_P = -167.

    # # geodetic coordinates
    # h = np.array([h_P, h_P+200, h_P-300.])
    # lat = np.array([lat_P, lat_P, lat_P])
    # lon = np.array([lon_P, lon_P, lon_P])

    # # geocentric cartesian coordinates
    # X, Y, Z = tr.GGC2GCC(h, lat, lon, a, b)
    # X_P, Y_P, Z_P = tr.GGC2GCC(h_P, lat_P, lon_P, a, b)

    # # computed topocentric Cartesian coordinates
    # x, y, z = tr.GCC2TCC(X, Y, Z, X_P, Y_P, Z_P, lat_P, lon_P)

    # aae(z[0], 0, decimal=15)
    # aae(z[1], -200, decimal=9)
    # aae(z[2], 300, decimal=9)


#### unit vectors


def test_normal_unit_vector_entrada_invalida():
    'pontos que não podem ser convertidos para numpy array 2D de floats '

    # string
    pontos_invalidos = '1234'
    raises(ValueError, tr.normal_unit_vector, pontos_invalidos)

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(ValueError, tr.normal_unit_vector, pontos_invalidos)

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(ValueError, tr.normal_unit_vector, pontos_invalidos)


def test_normal_unit_vector_unit_lenght():
    "verify if the computed vector has unit lenght"

    # GG coordinates
    coordinates = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0,    0,   90,  90,   90,  180,    0]
        ]).T

    # computed unit vectors
    vector = tr.normal_unit_vector(points=coordinates)

    # Euclidean norms
    norm_vector = np.sqrt(np.sum(vector*vector, axis=1))

    aae(norm_vector, np.ones(coordinates.shape[0]), decimal=10)


def test_latitude_unit_vector_unit_lenght():
    "verify if the computed vector has unit lenght"

    # GG coordinates
    coordinates = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0, -130,   90,  90,   90,  180,    0]
        ]).T

    # computed unit vectors
    vector = tr.latitude_unit_vector(points=coordinates)

    # Euclidean norms
    norm_vector = np.sqrt(np.sum(vector*vector, axis=1))

    aae(norm_vector, np.ones(coordinates.shape[0]), decimal=10)


def test_longitude_unit_vector_unit_lenght():
    "verify if the computed vector has unit lenght"

    # GG coordinates
    coordinates = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0, -130,   90, -90,   90,  180,    0]
        ]).T

    # computed unit vectors
    vector = tr.longitude_unit_vector(points=coordinates)

    # Euclidean norms
    norm_vector = np.sqrt(np.sum(vector*vector, axis=1))

    aae(norm_vector, np.ones(coordinates.shape[0]), decimal=10)