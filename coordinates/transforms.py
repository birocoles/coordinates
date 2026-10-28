'''
Python package to manipulate coordinate transformations commonly used in
Geophysics. They are:

*  Gencentric Cartesian (GC) Coordinates
*  Geocentric Spherical (GS) Coordinates
*  Geocentric  Geodetic (GG) Coordinates
* Topocentric Cartesian (TC) Coordinates
'''

import numpy as np


#### Auxiliary functions


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


def check_points(obj, latlon=False):
    '''
    Verify if 'obj' is a 2D numpy array of floats with 3 columns.
    If 'latlon' is True, force values in the second and third columns
    to be in the intervals [-90°, 90] and [-180°, 180], respectively.
    Default is latlon = False.


    parameters
    ----------
    obj : Python object to be verified

    latlon : boolean
        If True, force values in the second and third columns to be in 
        the intervals [-90°, 90] and [-180°, 180], respectively.

    returns
    -------
    P : numpy array 2D
        The numpy array representing a finite set of points.
        If 'obj' cannot be converted to a 2D numpy array of floats,
        then the function raises a ValueError.
    '''

    try:
        P = np.asarray(obj, dtype=float)
    except:
        raise ValueError("object must be convertible to numpy array")
    if P.ndim != 2:
        raise ValueError("object must be a 2D numpy array")
    if P.shape[1] != 3:
        raise ValueError("object must have 3 columns")
    if latlon is True:
        if np.any(P[:,1] > 90.0) or np.any(P[:,1] < -90.0):
            raise ValueError("latitude values in 2nd column must be in the interval [-90°, 90°]")
        if np.any(P[:,2] > 180.0) or np.any(P[:,2] < -180.0):
            raise ValueError("longitude values in the 3rd column must be in the interval [-180°, 180°]")

    return P


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
        Vector with geodetic latitude values (in degrees). This function
        presumes that the latitude values are in the interval [-90, 90].

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

    # Get the datum parameters
    a, f = elipsoid(datum=datum)

    # squared first eccentricity
    e2 = squared_first_eccentricity(datum=datum)

    # squared sine of latitude
    sin2lat = np.sin(np.deg2rad(lat))
    sin2lat *= sin2lat

    N = a / np.sqrt(1. - e2 * sin2lat)

    return N


#### Coordinate transformations


def GG2GC(points, datum):
    '''
    Transform GG into GC coordinates according to:

    X = (N + h) cos(lat) cos(lon)
    Y = (N + h) cos(lat) sin(lon)
    Z = [N (1 - e^2) + h] sin(lat)

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GG coordinates geometric altitude (in meters), geodetic latitude (in degrees) 
        and longitude (in degrees).
    datum: string
        Geodetic datum (see function 'elipsoid').

    returns
    -------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.
    '''

    P = check_points(obj=points, latlon=True)

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


def GC2GG(points, datum, itmax=5):
    '''
    Transform GC coordinates into GG coordinates using the Hirvonen-Moritz algorithm
    (Hofmann-Wellenhof and Moritz, 2005, p. 193).

    references
    ----------
    * Hofmann-Wellenhof, Bernhard, and Helmut Moritz. 2005. Physical Geodesy. Springer.

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
        the GG coordinates geometric altitude (in meters), geodetic latitude (in degrees) and 
        longitude (in degrees).
    '''

    P = check_points(points)
    if isinstance(itmax, int) is False:
        raise ValueError("itmax must be an integer")
    if itmax <= 0:
        raise ValueError("itmax must be positive")

    # Get the datum parameters
    a, f = elipsoid(datum=datum)
    b = a * (1.0 - f) # minor semiaxis

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


def GS2GC(points):
    '''
    Transform GS into GC coordinates according to:

    X = r cos(lat) cos(lon)
    Y = r cos(lat) sin(lon)
    Z = r sin(lat)

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GS coordinates radial distance (in meters), geocentric latitude (in degrees) and 
        longitude (in degrees).

    returns
    -------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.
    '''

    P = check_points(obj=points, latlon=True)

    # sine/cosine of latitude/longitude
    cos_lat = np.cos(np.deg2rad(P[:,1]))
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lat = np.sin(np.deg2rad(P[:,1]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    # X = r cos(lat) cos(lon)
    X = P[:,0] * cos_lat * cos_lon

    # Y = r cos(lat) sin(lon)
    Y = P[:,0] * cos_lat * sin_lon

    # Z = r sin(lat)
    Z = P[:,0] * sin_lat

    # transformed_points
    return np.vstack([X, Y, Z]).T


def GC2GS(points):
    '''
    Transform GC into GS coordinates according to:

    r   = sqrt( H^2 + Z^2 )
    lat = sin^(-1)( Z / r )
    lon = sin^(-1)( Y / H )

    where H = sqrt( X^2 + Y^2 ).

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.

    returns
    -------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GS coordinates radial distance (in meters), geocentric latitude (in degrees) and 
        longitude (in degrees).
    '''

    P = check_points(points)

    # H^2 = X^2 + Y^2
    H2 = P[:,0]**2 + P[:,1]**2 
    nonzero_H2 = (H2 >= 1e-7)
    # r   = sqrt( H^2 + Z^2 )
    r = np.sqrt(H2 + P[:,2]**2)
    # lat = sin^(-1)( Z / r )
    lat = np.empty_like(H2)
    lat[nonzero_H2] = np.rad2deg(np.arcsin(P[:,2][nonzero_H2] / r[nonzero_H2]))
    lat[~nonzero_H2] = np.sign(P[:,2][~nonzero_H2]) * 90.0
    # lon = sin^(-1)( Y / H )
    lon = np.empty_like(H2)
    # lon[nonzero_H2] = np.rad2deg(np.arcsin(P[:,1][nonzero_H2] / np.sqrt(H2[nonzero_H2])))
    lon[nonzero_H2] = np.rad2deg(np.arctan2(P[:,1][nonzero_H2], P[:,0][nonzero_H2]))
    lon[~nonzero_H2] = 0.0 # Assumed longitude value for the case in which latitude is +/-90°

    # transformed_points
    return np.vstack([r, lat, lon]).T


def GC2TC(points, origin, datum):
    '''
    Convert GC coordinates into TC coordinates with predefined origin. 
    The origin is a point defined in the Geocentric Geodetic Coordinate System. 
    The axes of the Topocentric Cartesian Coordinate System has axes x, y and z pointing 
    to, respectively, north, east and down (normal to a reference elipsoid). 
    
    The transformation is computed as follows:

    x =  vX*(X - X0) + vY*(Y - Y0) + vZ*(Z - Z0)
    y =  wX*(X - X0) + wY*(Y - Y0)
    z = -uX*(X - X0) - uY*(Y - Y0) - uZ*(Z - Z0)

    where uX, uY, uZ, vX, vY, vZ, wX, and wy are components of the unit vectors
    forming the local basis of the Geocentric Geodetic Coordinate system at the
    predefined origin.

    Parameters:
    -----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the GC coordinates X, Y and Z, in meters.
    origin : 3-tuple or 3-list
        GG coordinates geometric altitude, geodetic latitude and longitude of
        the origin.
    datum: string
        Geodetic datum (see function 'elipsoid') of the origin.

    Returns:
    --------
    transformed_points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the TC coordinates x, y and z, in meters.
    '''

    P  = check_points(points)
    h0, lat0, lon0 = origin
    P0_geodetic = np.array([[h0, lat0, lon0],])
    P0_Cartesian = GG2GC(points=P0_geodetic, datum=datum)
    
    # Differences in Geocentric Cartesian Coordinates
    DX = P[:,0] - P0_Cartesian[0,0]
    DY = P[:,1] - P0_Cartesian[0,1]
    DZ = P[:,2] - P0_Cartesian[0,2]

    # Unit vectors pointing to the orthogonal directions of the
    # Geocentric Geodetic System at the origin Q
    u =    normal_unit_vector(P0_geodetic)
    v =  latitude_unit_vector(P0_geodetic)
    w = longitude_unit_vector(P0_geodetic)

    # Coordinates in the Topocentric Cartesian System with origin at Q
    x =  v[0,0]*DX + v[0,1]*DY + v[0,2]*DZ
    y =  w[0,0]*DX + w[0,1]*DY
    z = -u[0,0]*DX - u[0,1]*DY - u[0,2]*DZ

    # transformed_points
    return np.vstack([x, y, z]).T


#### unit vectors


def normal_unit_vector(points):
    '''
    Compute the elements of the normal unit vector that points outward. 
    If 'points' contains GG coordinates, then the computed vector is normal 
    to the reference elipsoid. If 'points' contains GS coordinates, then the 
    computed vector is normal to the sphere that passes at the computation 
    points.
    
    At each point (each row of 'points'), the vector elements are given by:

        | cos(lat)*cos(lon) |
    u = | cos(lat)*sin(lon) | .
        | sin(lat)          |

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the (i) GG coordinates geometric altitude (in meters), geodetic latitude (in degrees) 
        and longitude (in degrees) or (ii) GS coordinates radial distance (in meters), 
        geocentric latitude (in degrees) and longitude (in degrees).
        Latitude values must be in the interval [-90°, 90°],
        Longitude values must be in the interval [-180°, 180°].

    returns
    -------
    vector_u: numpy array 2D
        Matrix with first, second and third columns formed by the Cartesian components
        uX, uY and uZ along the X, Y and Z axes of the Geocentric Cartesian Coordinate
        System at each point (each line of 'points').

    '''
    
    P = check_points(obj=points, latlon=True)

    # sine/cosine of the latitude/longitude
    cos_lat = np.cos(np.deg2rad(P[:,1]))
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lat = np.sin(np.deg2rad(P[:,1]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    vetor_u = np.vstack([
        cos_lat * cos_lon, cos_lat * sin_lon, sin_lat
        ]).T

    return vetor_u


def latitude_unit_vector(points):
    '''
    Compute the elements of the unit vector that points along the direction
    of increasing latitude. 
    If 'points' contains GG coordinates, then the computed vector is tangent 
    to the reference elipsoid. If 'points' contains GS coordinates, then the 
    computed vector is tangent to the sphere that passes at the computation 
    points.
    
    At each point (each row of 'points'), the vector elements are given by:

        | -sin(lat)*cos(lon) |
    v = | -sin(lat)*sin(lon) | .
        |  cos(lat)          |

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the (i) GG coordinates geometric altitude (in meters), geodetic latitude (in degrees) 
        and longitude (in degrees) or (ii) GS coordinates radial distance (in meters), 
        geocentric latitude (in degrees) and longitude (in degrees).
        Latitude values must be in the interval [-90°, 90°],
        Longitude values must be in the interval [-180°, 180°].

    returns
    -------
    vector_v: numpy array 2D
        Matrix with first, second and third columns formed by the Cartesian components
        vX, vY and vZ along the X, Y and Z axes of the Geocentric Cartesian Coordinate
        System at each point (each line of 'points').

    '''

    P = check_points(obj=points, latlon=True)

    # sine/cosine of the latitude/longitude
    cos_lat = np.cos(np.deg2rad(P[:,1]))
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lat = np.sin(np.deg2rad(P[:,1]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    vetor_v = np.vstack([
        -sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat
        ]).T

    return vetor_v


def longitude_unit_vector(points):
    '''
    Compute the elements of the unit vector that points along the direction
    of increasing longitude. 
    If 'points' contains GG coordinates, then the computed vector is tangent 
    to the reference elipsoid. If 'points' contains GS coordinates, then the 
    computed vector is tangent to the sphere that passes at the computation 
    points.
    
    At each point (each row of 'points'), the vector elements are given by:

        | -sin(lon) |
    w = |  cos(lon) | .
        |     0     |

    parameters
    ----------
    points : numpy array 2D
        Matrix with first, second and third columns formed, respectively, by 
        the (i) GG coordinates geometric altitude (in meters), geodetic latitude (in degrees) 
        and longitude (in degrees) or (ii) GS coordinates radial distance (in meters), 
        geocentric latitude (in degrees) and longitude (in degrees).
        Latitude values must be in the interval [-90°, 90°],
        Longitude values must be in the interval [-180°, 180°].

    returns
    -------
    vector_w: numpy array 2D
        Matrix with first, second and third columns formed by the Cartesian components
        wX and wY (wZ is always zero) along the X and Y axes of the Geocentric Cartesian Coordinate
        System at each point (each line of 'points').

    '''

    P = check_points(obj=points, latlon=True)

    # sine/cosine of the longitude
    cos_lon = np.cos(np.deg2rad(P[:,2]))
    sin_lon = np.sin(np.deg2rad(P[:,2]))

    vetor_w = np.vstack([
        -sin_lon, cos_lon
        ]).T

    return vetor_w