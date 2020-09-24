def WGS84():
    '''
    This function returns the following parameters defining the
    reference elipsoid WGS84:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378137.0
    f = 1/298.257223563

    return a, f


def GRS80():
    '''
    This function returns the following parameters defining the
    reference elipsoid GRS80:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378137.0
    f = 1/298.257222101

    return a, f


def SAD69():
    '''
    This function returns the following parameters defining the
    reference elipsoid SAD69:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378160.0
    f = 1.0/298.25

    return a, f

def Hayford1924():
    '''
    This function returns the following parameters defining the
    International (Hayford's) reference elipsoid of 1924:
    a = semimajor axis [m]
    f = flattening

    output:
    a, f
    '''
    a = 6378388.0
    f = 1.0/297.0

    return a, f


def WGS84_SAD69():
    '''
    Transformation parameters from local geodetic system
    WGS84 to SAD69.

    output
    dx: float - origin translation along the x-axis (in meters).
    dy: float - origin translation along the y-axis (in meters).
    dz: float - origin translation along the z-axis (in meters).
    '''

    dx = 66.87
    dy = -4.37
    dz = 38.52

    return dx, dy, dz


def SAD69_GRS80():
    '''
    Transformation parameters from local geodetic system
    SAD69 to GRS80.

    output
    dx: float - origin translation along the x-axis (in meters).
    dy: float - origin translation along the y-axis (in meters).
    dz: float - origin translation along the z-axis (in meters).
    '''

    dx = -67.35
    dy = 3.88
    dz = -38.22

    return dx, dy, dz
