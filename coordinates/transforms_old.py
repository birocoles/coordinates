'''
Collection of algorithms for transforming coordinates commoly used in
Geophysics.

Glossary:
---------
Geocentric Geodetic Coordinates (GGC)
Geocentric Geodetic System (GGS)
Geocentric Cartesian Coordinates (GCC)
Geocentric Cartesian System (GCS)
Geocentric Spherical Coordinates (GSC)
Geocentric Spherical System (GSS)
Topocentric Cartesian Coordinates (TCC)
Topocentric Cartesian System (TCS)

'''

import numpy as np
from . import utils

def GGC2GCC(height, latitude, longitude, major_semiaxis, minor_semiaxis):
    '''
    Transform GGC into GCC.

    Parameters:
    -----------
    height: numpy array 1D
        Vector containing the geometric height (in meters).
    latitude: numpy array 1D
        Vector containing the latitude (in degrees).
    longitude: numpy array 1D
        Vector containing the longitude (in degrees).
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).

    Returns:
    --------
    X: numpy array 1D
        Vector containing the X component of the Cartesian coordinates (in
        meters).
    Y: numpy array 1D
        Vector containing the X component of the Cartesian coordinates (in
        meters).
    Z: numpy array 1D
        Vector containing the Z component of the Cartesian coordinates (in
        meters).
    '''

    h = np.asarray(height)
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)

    assert (h.size == lat.size == lon.size), 'height, latitude \
and longitude must have the same number of elements'
    assert (major_semiaxis > minor_semiaxis), 'major_semiaxis must be greater \
than the minor_semiaxis'

    #Prime vertical radius of curvature
    N = utils.prime_vertical_curv(major_semiaxis, minor_semiaxis, lat)

    # convert degrees to radians
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    aux = N + height

    # squared first eccentricity
    e2 = (major_semiaxis**2. - minor_semiaxis**2.)/(major_semiaxis**2.)

    clat = np.cos(lat)
    slat = np.sin(lat)
    clon = np.cos(lon)
    slon = np.sin(lon)

    X = aux*clat*clon
    Y = aux*clat*slon
    Z = (N*(1 - e2) + height)*slat

    return X, Y, Z


def GCC2GGC(X, Y, Z, major_semiaxis, minor_semiaxis, itmax = 5):
    '''
    Convert GCC into GGC by using the Hirvonen-Moritz algorithm
    (Hofmann-Wellenhof and Moritz, 2005, p. 193).

    Parameters:
    -----------
    X: numpy array 1D or float
        Vector containing the x component of the Cartesian coordinates (in
        meters).
    Y: numpy array 1D or float
        Vector containing the y component of the Cartesian coordinates (in
        meters).
    Z: numpy array 1D or float
        Vector containing the z component of the Cartesian coordinates (in
        meters).
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).
    itmax: int
        Maximum number of iterations in the Hirvonen-Moritz algorithm. Default
        is 5.

    Returns:
    --------
    height: numpy array 1D
        Vector containing the geometric height (in meters).
    latitude: numpy array 1D
        Vector containing the latitude (in degrees).
    longitude: numpy array 1D
        Vector containing the longitude (in degrees).
    '''

    x = np.asarray(X)
    y = np.asarray(Y)
    z = np.asarray(Z)

    assert (x.size == y.size == z.size), \
'x, y and z must have the same number of elements'
    assert (major_semiaxis > minor_semiaxis), 'major_semiaxis must be greater \
than the minor_semiaxis'

    # horizontal distance
    p = np.sqrt(x**2. + y**2.)

    # null and non-null horizontal distances
    p_non_null = (p >= 1e-8)
    p_null = np.logical_not(p_non_null)

    lon = np.zeros_like(x)
    lat = np.zeros_like(x)
    height = np.zeros_like(x)

    # squared first eccentricity
    e2 = (major_semiaxis**2. - minor_semiaxis**2.)/(major_semiaxis**2.)

    aux1 = z[p_non_null]/p[p_non_null]
    aux2 = 1.- e2

    # define the coordinates for null horizontal distances
    lon[p_null] = 0.
    height[p_null] = np.abs(z[p_null]) - minor_semiaxis
    lat[p_null] = np.sign(z[p_null])*np.pi*0.5

    # first iteration
    lat[p_non_null] = np.arctan(aux1/aux2)
    sinlat = np.sin(lat[p_non_null])
    N = major_semiaxis/np.sqrt(1 - e2*sinlat*sinlat)
    height[p_non_null] = p[p_non_null]/np.cos(lat[p_non_null]) - N

    for i in range(itmax):
        aux3 = e2*N/(N + height[p_non_null])
        lat[p_non_null] = np.arctan(aux1/(1.-aux3))
        sinlat = np.sin(lat[p_non_null])
        N = major_semiaxis/np.sqrt(1 - e2*sinlat*sinlat)
        height[p_non_null] = p[p_non_null]/np.cos(lat[p_non_null]) - N

    lon[p_non_null] = np.arctan2(y[p_non_null], x[p_non_null])

    # convert latitude and longitude from radians to degrees
    latitude = np.rad2deg(lat)
    longitude = np.rad2deg(lon)

    return height, latitude, longitude


def GCC2GGC_approx(X, Y, Z, major_semiaxis, minor_semiaxis):
    '''
    Convert GCC into GGC by using an approximated formula (Hofmann-Wellenhof
    and Moritz, 2005, p. 196).

    Parameters:
    -----------
    X: numpy array 1D or float
        Vector containing the x component of the Cartesian coordinates (in
        meters).
    Y: numpy array 1D or float
        Vector containing the y component of the Cartesian coordinates (in
        meters).
    Z: numpy array 1D or float
        Vector containing the z component of the Cartesian coordinates (in
        meters).
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).

    Returns:
    --------
    height: numpy array 1D
        Vector containing the geometric height (in meters).
    latitude: numpy array 1D
        Vector containing the latitude (in degrees).
    longitude: numpy array 1D
        Vector containing the longitude (in degrees).

    '''

    x = np.asarray(X)
    y = np.asarray(Y)
    z = np.asarray(Z)

    assert (x.size == y.size == z.size), \
'x, y and z must have the same number of elements'
    assert (major_semiaxis > minor_semiaxis), 'major_semiaxis must be greater \
than the minor_semiaxis'

    # horizontal distance
    p = np.sqrt(x**2. + y**2.)

    # null and non-null horizontal distances
    p_non_null = (p >= 1e-8)
    p_null = np.logical_not(p_non_null)

    lon = np.zeros_like(x)
    lat = np.zeros_like(x)
    height = np.zeros_like(x)

    # define the coordinates for null horizontal distances
    lon[p_null] = 0.
    height[p_null] = np.abs(z[p_null]) - minor_semiaxis
    lat[p_null] = np.sign(z[p_null])*np.pi*0.5

    # squared first eccentricity
    e2 = (major_semiaxis**2. - minor_semiaxis**2.)/(major_semiaxis**2.)

    # squared second eccentricity
    elinha2 = (major_semiaxis**2. - minor_semiaxis**2.)/(minor_semiaxis**2.)

    # auxiliary variable
    theta = np.arctan(
        z[p_non_null]*major_semiaxis/(p[p_non_null]*minor_semiaxis)
    )
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    aux1 = z[p_non_null] + elinha2*minor_semiaxis*sintheta*sintheta*sintheta
    aux2 = p[p_non_null] - e2*major_semiaxis*costheta*costheta*costheta

    #lat[p_non_null] = np.arctan(aux1/aux2)
    lat[p_non_null] = np.arctan2(aux1, aux2)
    #lon[p_non_null] = np.arctan(y[p_non_null]/x[p_non_null])
    lon[p_non_null] = np.arctan2(y[p_non_null], x[p_non_null])

    sinlat = np.sin(lat[p_non_null])
    N = major_semiaxis/np.sqrt(1 - e2*sinlat*sinlat)

    height[p_non_null] = p[p_non_null]/np.cos(lat[p_non_null]) - N

    # convert latitude and longitude from radians to degrees
    latitude = np.rad2deg(lat)
    longitude = np.rad2deg(lon)

    return height, latitude, longitude


def GCC2TCC(X, Y, Z, X0, Y0, Z0, latitude_0, longitude_0):
    '''
    Convert GCC into TCC with origin at a point Q = (X0, Y0, Z0). The point Q
    has latitude and longitude given by latitude_0 and longitude_0,
    respectively. This TCS has axis x pointing to north, axis z
    pointing to the inward normal and axis y completing the right-handed
    system. If latitude_0 is geodetic, then the computed normal is defined with
    respect to the referrence elipsoid. If latitude_0 is spherical, then the
    normal is defined with respect to a sphere.
    The transformation is computed as follows:

    x =  vX*(X - X0) + vY*(Y - Y0) + vZ*(Z - Z0)
    y =  wX*(X - X0) + wY*(Y - Y0)
    z = -uX*(X - X0) - uY*(Y - Y0) - uZ*(Z - Z0)

    where uX, uY, uZ, vX, vY, vZ, wX, and wy are components of the unit vectors
    (referred to the GCS) pointing to the orthogonal directions of the GGS
    at the point Q.

    Parameters:
    -----------
    X, Y, Z: numpy arrays 1D
        Vectors containing the coordinates x, y and z (in meters), respectively,
        of the points referred to the GCS.
    X0, Y0, Z0: floats
        Coordinates of the origin in the GCS.
    latitude_0: float
        Latitude (in degrees) of the origin Q.
    longitude_0: float
        Longitude (in degrees) of the origin Q.

    Returns:
    --------
    x: numpy array 1D
        Vector containing the x component of TCC (in meters).
    y: numpy array 1D
        Vector containing the y component of TCC (in meters).
    z: numpy array 1D
        Vector containing the z component of TCC (in meters).
    '''

    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    assert (X.shape == Y.shape == Z.shape), 'X, Y and Z must have the same \
shape'
    assert np.isscalar(X0), 'X0 must be a scalar'
    assert np.isscalar(Y0), 'Y0 must be a scalar'
    assert np.isscalar(Z0), 'Z0 must be a scalar'
    assert np.isscalar(latitude_0), 'latitude_0 must be a scalar'
    assert np.isscalar(longitude_0), 'longitude_0 must be a scalar'

    # Differences in Geocentric Cartesian Coordinates
    DX = X - X0
    DY = Y - Y0
    DZ = Z - Z0

    # Unit vectors pointing to the orthogonal directions of the
    # Geocentric Geodetic System at the origin Q
    uX, uY, uZ = utils.unit_vector_normal(latitude_0, longitude_0)
    vX, vY, vZ = utils.unit_vector_latitude(latitude_0, longitude_0)
    wX, wY = utils.unit_vector_longitude(longitude_0)

    # Coordinates in the Topocentric Cartesian System with origin at Q
    x =  vX*DX + vY*DY + vZ*DZ
    y =  wX*DX + wY*DY
    z = -uX*DX - uY*DY - uZ*DZ

    return x, y, z


def TCC2GCC(x, y, z, X0, Y0, Z0, latitude_0, longitude_0):
    '''
    Convert TCC with origin at a point Q into GCC. The point Q has GCC
    coordinates (X0, Y0, Z0) and also the coordinates latitude_0 and
    longitude_0. This TCS has axis x pointing to north, axis z pointing to the
    inward normal and axis y completing the right-handed system. If latitude_0
    is defined in the GGS, then the computed normal is defined with respect to
    the referrence elipsoid. If latitude_0 is defined in the GSS, then the
    normal is defined with respect to a sphere. The transformation is computed
    as follows:

    X = vX*x + wX*y - uX*z + X0
    Y = vY*x + wY*y - uY*z + Y0
    Z = vZ*x        - uZ*z + Z0

    where uX, uY, uZ, vX, vY, vZ, wX, and wy are components of the unit vectors
    (referred to the GCS) pointing to the orthogonal directions of the GGS at
    the point Q.

    Parameters:
    -----------
    x, y, z: numpy arrays 1D
        Vectors containing the coordinates x, y and z (in meters), respectively,
        of the points referred to the TCS.
    X0, Y0, Z0: floats
        Coordinates of the TGCS origin in the GCS.
    latitude_0: float
        Latitude (in degrees) of the origin Q.
    longitude_0: float
        Longitude (in degrees) of the origin Q.

    Returns:
    --------
    X, Y, Z: numpy arrays 1D
        Vectors containing the coordinates X, Y and Z (in meters), respectively,
        of the points referred to the GCS.
    '''

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    assert (x.size == y.size == z.size), 'x, y and z must have the same size'
    assert np.isscalar(X0), 'X0 must be a scalar'
    assert np.isscalar(Y0), 'Y0 must be a scalar'
    assert np.isscalar(Z0), 'Z0 must be a scalar'
    assert np.isscalar(latitude_0), 'latitude_0 must be a scalar'
    assert np.isscalar(longitude_0), 'longitude_0 must be a scalar'

    # Unit vectors pointing to the orthogonal directions of the
    # Geocentric Geodetic System at the origin Q
    uX, uY, uZ = utils.unit_vector_normal(latitude_0, longitude_0)
    vX, vY, vZ = utils.unit_vector_latitude(latitude_0, longitude_0)
    wX, wY = utils.unit_vector_longitude(longitude_0)

    # Coordinates in the GCS
    X = vX*x + wX*y - uX*z + X0
    Y = vY*x + wY*y - uY*z + Y0
    Z = vZ*x        - uZ*z + Z0

    return X, Y, Z


def GSC2GCC(radius, latitude, longitude):
    '''
    Transform GSC into GCC. For each point, the tranformation is given by:

    X = radius * cos(latitude) * cos(longitude)
    Y = radius * cos(latitude) * sin(longitude)
    Z = radius * sin(latitude)

    Parameters:
    -----------
    radius: numpy array 1D float
        Radial coordinates (in meters) to be transformed.
    latitude, longitude: numpy arrays 1D
        Spherical latitude and longitude (in degrees) to be transformed.

    Returns:
    --------
    X, Y, Z: numpy arrays 1D
        Computed Cartesian coordinates (in meters).

    '''
    radius = np.asarray(radius)
    latitude = np.asarray(latitude)
    longitude = np.asarray(longitude)

    assert radius.size == latitude.size == longitude.size, 'radius, latitude \
and longitude must have the same number of elements'
    latitude_max = np.max(latitude)
    latitude_min = np.min(latitude)
    assert latitude_max <= 90, 'maximum latitude must be <= 90 degrees'
    assert latitude_min >= -90, 'minimum latitude must be >= -90 degrees'

    # convert degrees to radian
    lat = np.deg2rad(latitude)
    lon = np.deg2rad(longitude)

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    X = radius*cos_lat*cos_lon
    Y = radius*cos_lat*sin_lon
    Z = radius*sin_lat

    return X, Y, Z


def GCC2GSC(X, Y, Z):
    '''
    Transform GCC into GSC. For each point, the tranformation is given by:

    radius = sqrt(X**2 + Y**2 + Z**2)
    latitude = arcsin(Z/sqrt(X**2 + Y**2 + Z**2))
    longitude = arctan(Y/X)

    Parameters:
    -----------
    X, Y, Z: numpy arrays 1D
        Cartesian coordinates (in meters) to be transformed.

    Returns:
    --------
    radius: numpy array 1D float
        Computed radial coordinates (in meters).
    latitude, longitude: numpy arrays 1D
        Computed spherical latitude and longitude (in degrees).

    '''
    X = np.asarray(X)
    Y = np.asarray(Y)
    Z = np.asarray(Z)

    assert X.size == Y.size == Z.size, 'X, Y and Z must have the same number \
of elements'

    # squared horizontal component
    h2 = X*X + Y*Y

    radius = np.sqrt(h2 + Z*Z)
    latitude = np.rad2deg(np.arcsin(Z/radius))
    longitude = np.rad2deg(np.arctan2(Y, X))

    return radius, latitude, longitude


def geodetic2geocentric_latitude(geodetic_latitude, major_semiaxis,
                                 minor_semiaxis):
    '''
    Transform geodetic into geocentric latitude.

    Parameters:
    -----------
    geodetic_latitude: numpy array 1D or float
        Geodetic latitude (in degrees) to be transformed.
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).

    Returns:
    --------
    geocentric_latitude: numpy array 1D or float
        Computed geocentric latitude (in degrees).
    '''

    assert np.isscalar(major_semiaxis), 'major_semiaxis must be a scalar'
    assert np.isscalar(minor_semiaxis), 'minor_semiaxis must be a scalar'
    assert (major_semiaxis > minor_semiaxis), 'major_semiaxis must be greater \
than the minor_semiaxis'

    latitude = np.deg2rad(geodetic_latitude)
    aux = (minor_semiaxis*minor_semiaxis)/(major_semiaxis*major_semiaxis)
    geocentric_latitude = np.rad2deg(np.arctan(aux*np.tan(latitude)))

    return geocentric_latitude


def geocentric2geodetic_latitude(geocentric_latitude, major_semiaxis,
                                 minor_semiaxis):
    '''
    Transform geocentric into geodetic latitude.

    Parameters:
    -----------
    geocentric_latitude: numpy array 1D or float
        Geocentric latitude (in degrees) to be transformed.
    major_semiaxis: float
        Major semiaxis of the reference ellipsoid (in meters).
    minor_semiaxis: float
        Minor semiaxis of the reference ellipsoid (in meters).

    Returns:
    --------
    geodetic_latitude: numpy array 1D or float
        Computed geodetic latitude (in degrees).
    '''

    assert np.isscalar(major_semiaxis), 'major_semiaxis must be a scalar'
    assert np.isscalar(minor_semiaxis), 'minor_semiaxis must be a scalar'
    assert (major_semiaxis > minor_semiaxis), 'major_semiaxis must be greater \
than the minor_semiaxis'

    latitude = np.deg2rad(geocentric_latitude)
    aux = (minor_semiaxis*minor_semiaxis)/(major_semiaxis*major_semiaxis)
    geodetic_latitude = np.rad2deg(np.arctan(np.tan(latitude)/aux))

    return geodetic_latitude


def molodensky_complete(
    height1, latitude1, longitude1,
    major_semiaxis1, flattening1, major_semiaxis2, flattening2,
    dX, dY, dZ
    ):
    '''
    Transform GGC (height1, latitude1, longitude1) referred to an ellispoid 1
    into GGC (height2, latitude2, longitude2) referred to an ellispoid 2 by
    using the complete Molodensky's formulas (Rapp, 1993, p. 79).

    Parameters:
    -----------
    height1: numpy array 1D
        Vector containing heights referred to ellipsoid 1 (in meters).
    latitude1: numpy array 1D
        Vector containing latitudes referred to ellipsoid 1 (in degrees).
    longitude1: numpy array 1D
        Vector containing longitudes referred to ellipsoid 1 (in degrees).
    major_semiaxis1: float
        Major semiaxis of ellipsoid 1 (in meters).
    flattening1: float
        Flattening of ellipsoid 1 (in meters).
    major_semiaxis2: float
        Major semiaxis of ellipsoid 2 (in meters).
    flattening2: float
        Flattening of ellipsoid 2 (in meters).
    dX: float
        Translation of ellipsoid 2 with relative to ellipsoid 1 (in meters)
        along X axis of GCS.
    dY: float
        Translation of ellipsoid 2 with relative to ellipsoid 1 (in meters)
        along Y axis of GCS.
    dZ: float
        Translation of ellipsoid 2 with relative to ellipsoid 1 (in meters)
        along Z axis of GCS.

    Returns:
    --------
    height2: numpy array 1D
        Vector containing heights referred to ellipsoid 2 (in meters).
    latitude2: numpy array 1D
        Vector containing latitudes referred to ellipsoid 2 (in degrees).
    longitude2: numpy array 1D
        Vector containing longitudes referred to ellipsoid 2 (in degrees).
    '''

    assert (height1.size == latitude1.size == longitude1.size), \
        'height1, latitude1, andlongitude1 must have the same size'
    assert (flattening1 < major_semiaxis1), 'flattening1 must be smaller than \
major_semiaxis1'
    assert (flattening2 < major_semiaxis2), 'flattening2 must be smaller than \
major_semiaxis2'

    sinlat = np.sin(np.deg2rad(latitude1))
    coslat = np.cos(np.deg2rad(latitude1))

    sinlon = np.sin(np.deg2rad(longitude1))
    coslon = np.cos(np.deg2rad(longitude1))

    da = major_semiaxis2 - major_semiaxis1
    df = flattening2 - flattening1
    a = major_semiaxis1 + 0.5*da
    f = flattening1 + 0.5*df
    b = (1. - f)*a
    e2 = (a**2. - b**2.)/a**2.

    W = np.sqrt(1. - e2*(sinlat**2.))
    W[W < 1e-10] == 1e-10
    M = a*(1. - e2)/(W**3.)
    N = a/W

    dlat_dX = -sinlat*coslon
    dlat_dY = -sinlat*sinlon
    dlat_dZ = coslat
    dlat_da = e2*sinlat*coslat/W
    dlat_df = sinlat*coslat*(M*a/b + N*b/a)
    dlon_dX = -sinlon
    dlon_dY = coslon
    dh_dX = coslat*coslon
    dh_dY = coslat*sinlon
    dh_dZ = sinlat
    dh_da = -W
    dh_df = a*(1. - f)*(sinlat**2.)/W

    dlat = (dlat_dX*dX + dlat_dY*dY + dlat_dZ*dZ + dlat_da*da + dlat_df*df)
    dlat /= (M + height1)
    dlon = (dlon_dX*dX + dlon_dY*dY)/((N + height1)*coslat)
    dh = dh_dX*dX + dh_dY*dY + dh_dZ*dZ + dh_da*da + dh_df*df

    height2 = height1 + dh
    latitude2 = latitude1 + dlat
    longitude2 = longitude1 + dlon

    return height2, latitude2, longitude2
