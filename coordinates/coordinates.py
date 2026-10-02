'''
Python package to manipulate coordinate transformations commonly used in
Geophysics. They are:

* Gencentric Cartesian (GC) Coordinates
* Geocentric Spherical (GS) Coordinates
* Geocentric Geodetic (GG) Coordinates
* Topocentric Cartesian (TC) Coordinates
'''

import numpy as np


def elipsoid(datum='WGS84'):
    '''
    Return parameters associated with different reference elipsoids. 

    parameters
    ----------
    datum : string
        Reference elipsoid associated to a geodetic datum. The possible elipsoids are: 
        'WGS84', 'GRS80', 'SAD69'. The default is 'WGS84'.

    returns
    -------
    a: float
        Major semiaxis (in meters).
    f: float
        Flattening (dimensionless).
    '''

    if isinstance(datum, str) is False:
        raise ValueError("datum must be a string")
    if datum not in ['WGS84', 'GRS80', 'SAD69']:
        raise ValueError("invalid datum")
    
    parameters = {
        'WGS84': (6378137.0, 1.0/298.257223563),
        'GRS80': (6378137.0, 1.0/298.257222101),
        'SAD69': (6378160.0, 1.0/298.25)
    }

    return parameters[datum]


def squared_first_eccentricity(datum='WGS84'):
    '''
    Return the squared first eccentricity for a given elipsoid. 

    parameters
    ----------
    datum : string
        Reference elipsoid associated to a geodetic datum. The possible elipsoids are: 
        'WGS84', 'GRS80', 'SAD69'. The default is 'WGS84'.

    returns
    -------
    e2 :  float
        Computed squared first eccentricity.
    '''

    # Get the datum parameters
    a, f = elipsoid(datum=datum)
    b = a * (1.0 - f) # minor semiaxis

    # squared first eccentricity
    a2 = a*a
    b2 = b*b
    e2 = (a2 - b2)/a2

    return e2


def prime_vertical_curv(geodetic_latitude, datum):
    '''
    Compute the prime vertical radius of curvature given by

    N = a / sqrt(1 - e^2 sin^2(lat)) ,

    where

    e^2 = (a^2 - b^2) / b^2,

    is the squared first eccentricity, 'a' is the major semiaxis and 'b = a (1 - f)' 
    is the minor semiaxis of the reference elipsoid.

    parameters
    ----------
    geodetic_latitude: numpy array 1D
        Vector with geodetic latitude values (in degrees).
    datum: string
        Geodetic datum (see function 'elipsoid').

    returns
    -------
    N: numpy array 1D
        Vector with the computed prime vertical radius of curvature.
    '''

    try:
        lat = np.asarray(geodetic_latitude, dtype=float)
    except:
        raise ValueError("geodetic_latitude must be convertible to numpy array")
    if lat.ndim != 1:
        raise ValueError("geodetic_latitude must be a 1D numpy array")

    # squared first eccentricity
    e2 = squared_first_eccentricity(datum=datum)

    # squared sine of latitude
    sin2lat = np.sin(np.deg2rad(lat))
    sin2lat *= sin2lat

    N = a / np.sqrt(1. - e2 * sin2lat)

    return N



def GG2GC(poits, datum):
    '''
    Transform GG into GC according to:

    X = (N + h) cos(lat) cos(lon)
    Y = (N + h) cos(lat) sin(lon)
    Z = [N (1 - e^2) + h] sin(lat)

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GG coordinates altitude (in meters), latitude (in degrees) and longitude (in degrees).
    datum: string
        Geodetic datum (see function 'elipsoid').

    returns
    -------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.
    '''

    try:
        P = np.asarray(points, dtype=float)
    except:
        raise ValueError("points must be convertible to numpy array")
    if P.ndim != 2:
        raise ValueError("points must be a 2D numpy array")
    if P.shape[1] != 3:
        raise ValueError("points must have 3 columns")

    # squared first eccentricity
    e2 = squared_first_eccentricity(datum=datum)

    # sine/cosine of latitude/longitude
    cos_lat = np.cos(np.deg2rad(P[:,1]))
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lat = np.sin(np.deg2rad(P[:,1]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    # prime vertical radius of curvature
    N = prime_vertical_curv(geodetic_latitude=P[:,1], datum=datum)

    # X = (N + h) cos(lat) cos(lon)
    X = (N + P[:,0]) * cos_lat * cos_lon

    # Y = (N + h) cos(lat) sin(lon)
    Y = (N + P[:,0]) * cos_lat * sin_lon

    # Z = [N (1 - e^2) + h] sin(lat)
    Z = (N * (1 - e2) + P[:,0]) * sin_lat

    # transformed_points
    return np.vstack([X, Y, Z]).T


def GC2GG(pontos, datum, itmax=5):
    '''
    Transform GC coordinates into GG coordinates using the Hirvonen-Moritz algorithm
    (Hofmann-Wellenhof and Moritz, 2005, p. 193).

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.
    datum: string
        Geodetic datum (see function 'elipsoid').
    itmax : int
        Positive integer defining the maximum number of iterations.
        The default is 5.


    returns
    -------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GG coordinates altitude (in meters), latitude (in degrees) and longitude (in degrees).
    '''

    try:
        P = np.asarray(points, dtype=float)
    except:
        raise ValueError("points must be convertible to numpy array")
    if P.ndim != 2:
        raise ValueError("points must be a 2D numpy array")
    if P.shape[1] != 3:
        raise ValueError("points must have 3 columns")
    if isinstance(itmax, int) is False:
        raise ValueError("itmax must be an integer")
    if itmax <= 0:
        raise ValueError("itmax must be positive")

    # squared first eccentricity
    e2 = squared_first_eccentricity(datum=datum)

    # horizontal distance
    p = np.sqrt(P[:,0]**2. + P[:,1]**2.)

    # null and non-null horizontal distance values
    p_non_null = (p >= 1e-8)
    p_null = np.logical_not(p_non_null)

    lon = np.zeros_like(P[:,0])
    lat = np.zeros_like(P[:,0])
    h = np.zeros_like(P[:,0])

    # auxiliary variables
    aux1 = P[:,2][p_non_null] / p[p_non_null]
    aux2 = 1.0 - e2

    # define coordinates for null horizontal distance values
    lon[p_null] = 0.
    h[p_null] = np.abs(P[:,2][p_null]) - b
    lat[p_null] = np.sign(P[:,2][p_null]) * np.pi * 0.5

    # first iteration
    lat[p_non_null] = np.arctan(aux1 / aux2)
    sinlat = np.sin(lat[p_non_null])
    N = a/np.sqrt(1.0 - e2 * sinlat * sinlat)
    h[p_non_null] = p[p_non_null] / np.cos(lat[p_non_null]) - N

    # remaining iterations
    for i in range(itmax):
        aux3 = e2 * N / (N + h[p_non_null])
        lat[p_non_null] = np.arctan(aux1 / (1.0 - aux3))
        sinlat = np.sin(lat[p_non_null])
        N = a / np.sqrt(1.0 - e2 * sinlat * sinlat)
        h[p_non_null] = p[p_non_null] / np.cos(lat[p_non_null]) - N

    lon[p_non_null] = np.arccos(P[:,0][p_non_null] / p[p_non_null])

    # convert values from radians to degrees
    latitude = np.rad2deg(lat)
    longitude = np.rad2deg(lon)

    # transformed_points
    return np.vstack([h, latitude, longitude]).T


def vector_normal_unitario(pontos):
    '''
    Calcula os elementos do vetor unitário u que aponta na direção normal 
    para fora. Se 'pontos' contém CGG, então o vetor u é normal ao elipsoide
    de referência. Se 'pontos' contém CEG, então o vetor u é normal a esfera.

    Em cada ponto (cada linha da matriz 'pontos'), as componentes de u são 
    dadas por:

        | cos(lat)*cos(lon) |
    u = | cos(lat)*sin(lon) | .
        | sin(lat)          |

    entrada
    -------
    pontos : numpy array 2D
        Matriz que contém CGG ou CEG. Se 'pontos' contém CGG, então sua 
        primeira, segunda e terceira colunas contém, respectivamente, 
        altitude geométrica (em metros), latitude geodésica (em graus) 
        e longitude (em graus). Se 'pontos' contém CEG, então sua 
        primeira, segunda e terceira colunas contém, respectivamente, 
        distância radial (em metros), latitude geocêntrica (em graus) 
        e longitude (em graus).

    saída
    -----
    vetor_u: numpy array 2D
        Matriz com primeira, segunda e terceira colunas formada pelas 
        componentes Cartesianas geocêntricas uX, uY e uZ do vetor unitário
        u em cada ponto (linha da matriz 'pontos').

    '''
    
    try:
        P = np.asarray(pontos, dtype=float)
    except:
        raise ValueError("pontos deve ser compatível com numpy array")
    if P.ndim != 2:
        raise ValueError("pontos deve ser um numpy array 2D")
    if P.shape[1] != 3:
        raise ValueError("pontos deve ter três colunas")

    # seno/cosseno da latitude/longitude
    cos_lat = np.cos(np.deg2rad(P[:,1]))
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lat = np.sin(np.deg2rad(P[:,1]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    vetor_u = np.vstack([
        cos_lat * cos_lon, cos_lat * sin_lon, sin_lat
        ]).T

    return vetor_u




