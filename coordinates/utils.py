import numpy as np

# Unit vectors


def unit_vector_normal(latitude, longitude):
    '''
    Compute the elements of a unit vector u pointing to the direction of
    the outward normal. If latitude is geodetic, then the computed u is
    normal to the referrence elipsoid. If latitude is spherical, then u is
    normal to the sphere. The vector u is referred to the GCS.
    At a given point with (spherical or geodetic) coordinates (lat, lon), the
    components of u along the axes X, Y and Z of a GCS can be written as follows:

        | cos(lat)*cos(lon) |
    u = | cos(lat)*sin(lon) | .
        | sin(lat)          |

    Parameters:
    -----------
    latitude: numpy array 1D
        Vector containing the latitude (in degrees) of the computation points.
    longitude: numpy array 1D
        Vector containing the lonitude (in degrees) of the computation points.

    Returns:
    --------
    uX, uY, uZ: numpy arrays 1D
        Components of the vector u.

    '''
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    assert latitude.size == longitude.size, 'latitude and longitude must have \
the same numer of elements'

    # convert degrees to radian
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)

    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)

    uX = coslat*coslon
    uY = coslat*sinlon
    uZ = sinlat

    return uX, uY, uZ


def unit_vector_latitude(latitude, longitude):
    '''
    Compute the elements of a unit vector v pointing to the direction of
    increasing latitude. If latitude is geodetic, then the computed v is
    tangent to the referrence elipsoid. If latitude is spherical, then v is
    tangent to the sphere. This vector is referred to the Geocentric
    Cartesian System. At a given point with (spherical or geodetic) coordinates
    (lat, lon), the components of v along the axes X, Y and Z of a Geocentric
    Cartesian System can be written as follows:

        | -sin(lat)*cos(lon) |
    v = | -sin(lat)*sin(lon) | .
        |  cos(lat)          |

    Parameters:
    -----------
    latitude: numpy array 1D
        Vector containing the latitude (in degrees) of the computation points.
    longitude: numpy array 1D
        Vector containing the lonitude (in degrees) of the computation points.

    Returns:
    --------
    vX, vY, vZ: numpy arrays 1D
        Components of the vector v.

    '''
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    assert latitude.size == longitude.size, 'latitude and longitude must have \
the same numer of elements'

    # convert degrees to radian
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)

    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)

    vX = -sinlat*coslon
    vY = -sinlat*sinlon
    vZ = coslat

    return vX, vY, vZ


def unit_vector_longitude(longitude):
    '''
    Compute the elements of a unit vector w pointing to the direction of
    increasing longitude. This vector is referred to the Geocentric
    Cartesian System. At a given point with (spherical or geodetic) coordinates
    (lat, lon), the components of w along the axes X, Y and Z of a Geocentric
    Cartesian System can be written as follows:

        | -sin(lon) |
    w = |  cos(lon) | .
        |  0        |

    Parameters:
    -----------
    longitude: numpy array 1D
        Vector containing the longitude (in degrees) of the computation points.

    Returns:
    --------
    wX, wY: numpy arrays 1D
        Non-null components of the vector w.

    '''
    longitude = np.asarray(longitude)

    # convert degrees to radian
    lon = np.deg2rad(longitude)

    coslon = np.cos(lon)
    sinlon = np.sin(lon)

    wX = -sinlon
    wY = coslon
    #wZ = 0.

    return wX, wY


def unit_vector_TCS(inclination, declination):
    '''
    Compute the components x, y and z of unit vectors v referred to a
    TCS. Each vector is defined by a pair (inclination, declination) as follows:
        | cos(inclination)*cos(declination) |
    v = | cos(inclination)*sin(declination) | .
        | sin(inclination)                  |

    Parameters:
    -----------
    inclination: numpy array 1D or float
        Inclination values (in degrees).
    declination: numpy array 1D or float
        Declination values (in degrees).

    Returns:
    --------
    vx, vy, vz: numpy arrays 1D or floats
        Components of the unit vector(s).

    '''
    inclination = np.asarray(inclination)
    declination = np.asarray(declination)

    assert inclination.size == declination.size, 'inclination and declination \
must have the same numer of elements'

    # convert degrees to radian
    inc = np.deg2rad(inclination)
    dec = np.deg2rad(declination)

    cosinc = np.cos(inc)
    sininc = np.sin(inc)
    cosdec = np.cos(dec)
    sindec = np.sin(dec)

    vx = cosinc*cosdec
    vy = cosinc*sindec
    vz = sininc

    return vx, vy, vz


# Rotation matrices

def rotation_NEU(latitude, longitude):
    '''
    Compute the elements of a rotation matrix whose columns are
    the unit vectors v (North), w (East) and u (Up). If latitude is geodetic,
    then the computed u is normal to the referrence elipsoid. If latitude is
    spherical, then u is normal to the sphere. The unit vectors v and w point
    to directions of increasing latitude (spherical or geodetic) and longitude,
    respectively. At a given point with coordinates (lat, lon), the rotation
    matrix can be written as follows:

        | -sin(lat)*cos(lon)   -sin(lon)    cos(lat)*cos(lon) |
    R = | -sin(lat)*sin(lon)    cos(lon)    cos(lat)*sin(lon) |
        |       cos(lat)           0             sin(lat)     |

    Parameters:
    -----------
    latitude: numpy array 1D
        Vector containing the latitude (in degrees) of the computation points.
    longitude: numpy array 1D
        Vector containing the lonitude (in degrees) of the computation points.

    Returns:
    --------
    R: numpy array 2D
        Matrix whose columns contain the elements 11, 12, 13, 21, 22, 23, 31,
        and 33 of the rotation matrix evaluated at the computation points.
        The element 32 is null and, consequently, it is not computed.

    '''
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    assert latitude.size == longitude.size, 'latitude and longitude must have \
the same numer of elements'

    # convert degrees to radian
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)

    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)

    R11 = -sinlat*coslon
    R21 = -sinlat*sinlon
    R31 = coslat

    R12 = -sinlon
    R22 = coslon

    R13 = coslat*coslon
    R23 = coslat*sinlon
    R33 = sinlat

    R = np.vstack([R11, R12, R13, R21, R22, R23, R31, R33]).T

    return R


def rotation_NED(latitude, longitude):
    '''
    Compute the elements of a rotation matrix whose columns are
    the unit vectors v (North), w (East) and -u (Down). If latitude is geodetic,
    then u is normal to the referrence elipsoid. If latitude is
    spherical, then u is normal to the sphere. The unit vectors v and w point
    to directions of increasing latitude (spherical or geodetic) and longitude,
    respectively. At a given point with coordinates (lat, lon), the rotation
    matrix can be written as follows:

        | -sin(lat)*cos(lon)   -sin(lon)   -cos(lat)*cos(lon) |
    R = | -sin(lat)*sin(lon)    cos(lon)   -cos(lat)*sin(lon) |
        |       cos(lat)           0            -sin(lat)     |

    Parameters:
    -----------
    latitude: numpy array 1D
        Vector containing the latitude (in degrees) of the computation points.
    longitude: numpy array 1D
        Vector containing the lonitude (in degrees) of the computation points.

    Returns:
    --------
    R: numpy array 2D
        Matrix whose columns contain the elements 11, 12, 13, 21, 22, 23, 31,
        and 33 of the rotation matrix evaluated at the computation points.
        The element 32 is null and, consequently, it is not computed.

    '''
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    assert latitude.size == longitude.size, 'latitude and longitude must have \
the same numer of elements'

    # convert degrees to radian
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)

    coslat = np.cos(lat)
    sinlat = np.sin(lat)
    coslon = np.cos(lon)
    sinlon = np.sin(lon)

    R11 = -sinlat*coslon
    R21 = -sinlat*sinlon
    R31 = coslat

    R12 = -sinlon
    R22 = coslon

    R13 = -coslat*coslon
    R23 = -coslat*sinlon
    R33 = -sinlat

    R = np.vstack([R11, R12, R13, R21, R22, R23, R31, R33]).T

    return R


def rotation_Rx(angle):
    '''
    Orthogonal matrix performing a rotation around
    the x-axis of a Cartesian coordinate system.

    Parameters:
    -----------
    angle : float
        Rotation angle (in degrees).

    Returns:
    --------
    R : 2D numpy array
        Rotation matrix.
    '''

    assert isinstance(1.*angle, float), 'angle must be a float'

    ang = np.deg2rad(angle)

    cos_angle = np.cos(ang)
    sin_angle = np.sin(ang)

    R = np.array([[1, 0, 0],
                  [0, cos_angle, sin_angle],
                  [0, -sin_angle, cos_angle]])

    return R


def rotation_Ry(angle):
    '''
    Orthogonal matrix performing a rotation around
    the y-axis of a Cartesian coordinate system.

    Parameters:
    -----------
    angle : float
        Rotation angle (in degrees).

    Returns:
    --------
    R : 2D numpy array
        Rotation matrix.
    '''

    assert isinstance(1.*angle, float), 'angle must be a float'

    ang = np.deg2rad(angle)

    cos_angle = np.cos(ang)
    sin_angle = np.sin(ang)

    R = np.array([[cos_angle, 0, -sin_angle],
                  [0, 1, 0],
                  [sin_angle, 0, cos_angle]])

    return R


def rotation_Rz(angle):
    '''
    Orthogonal matrix performing a rotation around
    the z-axis of a Cartesian coordinate system.

    Parameters:
    -----------
    angle : float
        Rotation angle (in degrees).

    Returns:
    --------
    R : 2D numpy array
        Rotation matrix.
    '''

    assert isinstance(1.*angle, float), 'angle must be a float'

    ang = np.deg2rad(angle)

    cos_angle = np.cos(ang)
    sin_angle = np.sin(ang)

    R = np.array([[cos_angle, sin_angle, 0],
                  [-sin_angle, cos_angle, 0],
                  [0, 0, 1]])

    return R


# Auxiliary quantities

def prime_vertical_curv(major_semiaxis, minor_semiaxis, latitude):
    '''
    Compute the prime vertical radius of curvature.

    Parameters:
    -----------
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).
    latitude: numpy array 1D
        Latitude (in degrees) of the computation points.

    output

    N: numpy array 1D
        Prime vertical radius of curvature computed at each latitude.
    '''

    assert major_semiaxis > minor_semiaxis, 'major_semiaxis must be greater \
than minor_semiaxis'
    assert major_semiaxis > 0, 'major semiaxis must be positive'
    assert minor_semiaxis > 0, 'minor semiaxis must be positive'

    # squared first eccentricity
    a2 = major_semiaxis*major_semiaxis
    b2 = minor_semiaxis*minor_semiaxis
    e2 = (a2 - b2)/a2

    # squared sine of latitude
    sin2lat = np.sin(np.deg2rad(latitude))
    sin2lat *= sin2lat

    N = major_semiaxis/np.sqrt(1. - e2*sin2lat)

    return N


def meridian_curv(major_semiaxis, minor_semiaxis, latitude):
    '''
    Compute the prime vertical radius of curvature.

    Parameters:
    -----------
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).
    latitude: numpy array 1D
        Latitude (in degrees) of the computation points.

    Returns:
    --------
    M: numpy array 1D
        Meridian radius of curvature computed at each latitude.
    '''

    assert major_semiaxis > minor_semiaxis, 'major_semiaxis must be greater \
than minor_semiaxis'
    assert major_semiaxis > 0, 'major semiaxis must be positive'
    assert minor_semiaxis > 0, 'minor semiaxis must be positive'

    # squared first eccentricity
    a2 = major_semiaxis*major_semiaxis
    b2 = minor_semiaxis*minor_semiaxis
    e2 = (a2 - b2)/a2

    # squared sine of latitude
    sin2lat = np.sin(np.deg2rad(latitude))
    sin2lat *= sin2lat

    # auxiliary variable
    aux = np.sqrt(1. - e2*sin2lat)

    M = (major_semiaxis*(1 - e2))/(aux*aux*aux)

    return M


def spherical_central_angle(lat1, lon1, lat2, lon2):
    '''
    Compute the central angle between two points located on a sphere
    with unit radius by using the special case of Vincenty formula
    (https://en.wikipedia.org/wiki/Great-circle_distance).

    Parameters:
    -----------
    lat1, lat2 : floats
        Latitude (in degrees) of the points 1 and 2.
    lon1, lon2 : floats
        Longitude (in degrees) of the points 1 and 2.

    Returns:
    --------
    central_angle : float
        Central angle in radians.
    '''
    lat1 = np.asarray(lat1)
    lat2 = np.asarray(lat2)
    lon1 = np.asarray(lon1)
    lon2 = np.asarray(lon2)
    slat1 = lat1.shape
    slat2 = lat2.shape
    slon1 = lon1.shape
    slon2 = lon2.shape
    assert slat1 == slat2 == slon1 == slon2, 'lat1, lon1, lat2 and lon2 must \
have the same shape'

    lat_rad1 = np.deg2rad(lat1)
    lat_rad2 = np.deg2rad(lat2)
    lon_rad1 = np.deg2rad(lon1)
    lon_rad2 = np.deg2rad(lon2)

    dlon = lon_rad2 - lon_rad1
    cosdlon = np.cos(dlon)
    sindlon = np.sin(dlon)
    coslat1 = np.cos(lat_rad1)
    coslat2 = np.cos(lat_rad2)
    sinlat1 = np.sin(lat_rad1)
    sinlat2 = np.sin(lat_rad2)

    aux1 = (coslat2*sindlon)**2
    aux2 = (coslat1*sinlat2 - sinlat1*coslat2*cosdlon)**2
    numerator = np.sqrt(aux1 + aux2)
    denominator = sinlat1*sinlat2 + coslat1*coslat2*cosdlon
    #central_angle = np.arctan2(numerator, denominator)
    central_angle = np.arctan(numerator/denominator)
    return central_angle
