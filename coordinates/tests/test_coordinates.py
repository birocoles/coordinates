import numpy as np
from numpy.testing import assert_almost_equal as aae
from pytest import raises
import coordenadas as co


def test_elipsoide_entrada_invalida():
    "datum inválido"

    # string inválida
    datum_invalida = '1234'
    raises(ValueError, co.elipsoide, datum_invalida)

    # inteiro
    datum_invalida = 12
    raises(ValueError, co.elipsoide, datum_invalida)

    # tupla
    datum_invalida = ('WGS84',)
    raises(ValueError, co.elipsoide, datum_invalida)

    # lista
    datum_invalida = ["GRS80",]
    raises(ValueError, co.elipsoide, datum_invalida)


def test_raio_curvatura_primeiro_vertical_entrada_invalida():
    "latitude que não pode ser convertida para array 1D de floats"

    datum_valido = "SAD69"

    # string
    latitude_invalida = '1234'
    raises(
        ValueError, 
        co.raio_curvatura_primeiro_vertical, 
        latitude_invalida,
        datum_valido
        )

    # lista com vários tipos
    latitude_invalida = np.array(['N45', 45, 45.0])
    raises(
        ValueError, 
        co.raio_curvatura_primeiro_vertical, 
        latitude_invalida,
        datum_valido
        )

    # matriz de floats
    latitude_invalida = np.ones((3,2))
    raises(
        ValueError, 
        co.raio_curvatura_primeiro_vertical, 
        latitude_invalida,
        datum_valido
        )


def test_CGG2CCG_entrada_invalida():
    'pontos que não podem ser convertidos para numpy array 2D de floats '

    datum_valido = "SAD69"

    # string
    pontos_invalidos = '1234'
    raises(
        ValueError, 
        co.CGG2CCG, 
        pontos_invalidos,
        datum_valido
        )

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(
        ValueError, 
        co.CGG2CCG, 
        pontos_invalidos,
        datum_valido
        )

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(
        ValueError, 
        co.CGG2CCG, 
        pontos_invalidos,
        datum_valido
        )


def test_CGG2CCG_pontos_conhecidos():
    'verifica coordenadas calculadas para entrada conhecida'

    # datum
    datum = "WGS84"

    # altitude, latitude e longitude
    pontos = np.array([
        [0, 0, 0], [0, 90, 0], [100.0, 0, 90]
        ])

    # Extrai os parâmetros do datum
    a, f = co.elipsoide(datum=datum)
    b = a * (1. - f) # semi-eixo menor

    # coordenadas verdadeiras
    coordenadas_verdadeiras = np.array([
        [a, 0., 0.], [0., 0., b], [0., a+100.0, 0.]
        ])

    # coordenadas calculadas
    coordenadas = co.CGG2CCG(pontos, datum)

    aae(coordenadas_verdadeiras, coordenadas, decimal=6)


def test_CCG2CGG_pontos_invalidos():
    'pontos que não podem ser convertidos para numpy array 2D de floats '

    datum_valido = "SAD69"

    # string
    pontos_invalidos = '1234'
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_invalidos,
        datum_valido
        )

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_invalidos,
        datum_valido
        )

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_invalidos,
        datum_valido
        )

def test_CCG2CGG_itmax_invalido():
    'itmax diferente de inteiro positivo'

    datum_valido = "SAD69"
    pontos_validos = np.ones((5,3))

    # string
    itmax_invalido = '1234'
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # list
    itmax_invalido = [3,]
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # inteiro negativo
    itmax_invalido = -3
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )

    # inteiro nulo
    itmax_invalido = 0
    raises(
        ValueError, 
        co.CCG2CGG, 
        pontos_validos,
        datum_valido,
        itmax_invalido
        )


def test_CGG2CCG_CCG2CGG():
    "compara coordenadas calculadas com verdadeiras"

    # coordenadas geodesicas verdadeiras
    coordenadas_verdadeiras = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0,    0,   90,  90,   90,  180,    0]
        ]).T

    # datum 
    datum = "WGS84"

    coordenadas_Cartesianas_calculadas = co.CGG2CCG(
        pontos=coordenadas_verdadeiras, 
        datum=datum
        )

    coordenadas_geodesicas_calculadas = co.CCG2CGG(
        pontos=coordenadas_Cartesianas_calculadas,
        datum=datum,
        itmax=5)

    aae(coordenadas_verdadeiras, coordenadas_geodesicas_calculadas, decimal=6)


def test_vector_normal_unitario_entrada_invalida():
    'pontos que não podem ser convertidos para numpy array 2D de floats '

    # string
    pontos_invalidos = '1234'
    raises(ValueError, co.vector_normal_unitario, pontos_invalidos)

    # lista
    pontos_invalidos = [34., -1, 0.4]
    raises(ValueError, co.vector_normal_unitario, pontos_invalidos)

    # array com num. de colunas diferente de 3
    pontos_invalidos = np.ones((3,2))
    raises(ValueError, co.vector_normal_unitario, pontos_invalidos)


def test_vector_normal_unitario_saida_unitaria():
    "verifica se a saída corresponde a vetores unitários"

    # coordenadas geodésicas geocêntricas (CGG)
    CGG = np.array([
        [0, -100,  200, -300, 400, -800, 1600, 7000],
        [0,  -15, 22.5,  -30,  45,  -60,   75,  -90],
        [0,    0,    0,   90,  90,   90,  180,    0]
        ]).T

    # vetores unitários calculados
    U = co.vector_normal_unitario(pontos=CGG)

    # norma Euclidiana dos vetores unitários
    norma_U = np.sqrt(np.sum(U*U, axis=1))

    aae(norma_U, np.ones(CGG.shape[0]), decimal=10)