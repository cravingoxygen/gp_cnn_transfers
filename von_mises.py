import numpy as np

def double_fac(m):
    i = 1
    result = 1
    for _ in range(m):
        result *= i
        i += 2
    return result

def standard_gaussian_moments(m):
    result = []
    for i in range(m):
        result.append(double_fac(i))
        result.append(0.)
    return np.array(result)

def atomic_approx(Mv):
    m = len(Mv)//2
    assert 2*m == len(Mv)

    #Construct matrix
    Mm = np.zeros([m, m])
    for i in range(m):
        for j in range(m):
            Mm[i, j] = Mv[i + j]

    #Check all determinants positive
    for i in range(m):
        assert 0 < np.linalg.det(Mm[:i, :i])

    #Compute c
    b = -Mv[-m:]
    # According to von-Misus
    c = np.linalg.solve(Mm, b)
    c = np.append(c, 1)
    c = np.flip(c, 0)
    av = np.roots(c)

    power = np.expand_dims(np.arange(m), 1)
    am = av ** power
    prob = np.linalg.solve(am, Mv[:m])

    #Test result
    power = np.expand_dims(np.arange(2*m), 1)
    E = (prob*av**power).sum(1)

    assert np.allclose(E, Mv)
    return (av, prob)

def standard_gaussian_atoms(m):
    return atomic_approx(standard_gaussian_moments(m))

a, prob = standard_gaussian_atoms(4)
