# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp
from scipy.sparse import csc_array, issparse

def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz jest diagonalnie zdominowana.

    Args:
        A (np.ndarray | sp.sparse.csc_array): Macierz A (m,m) podlegająca 
            weryfikacji.
    
    Returns:
        (bool): `True`, jeśli macierz jest diagonalnie zdominowana, 
            w przeciwnym wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    if not isinstance(A, (np.ndarray, csc_array)):
        return None

    if issparse(A):
        A = A.toarray()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    abs_A = np.abs(A)

    diag = np.diagonal(abs_A)
    row_sums = np.sum(abs_A, axis=1)
    off_diag_sums = row_sums - diag

    return np.all(diag > off_diag_sums)

   
    

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
            return None
        if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
            return None
        m, n = A.shape
        if len(x) != n or len(b) != m:
            return None
        return np.linalg.norm(A @ x - b)
    except Exception:
        return None
    return np.linalg.norm(A @ x - b)

