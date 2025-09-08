# Reference: https://github.com/YuvShap/finite-geometry-coverings-construction
import math
import numpy as np
import galois
from .gray_code import reduced_row_echelon_form_gray_code


class CoveringConstructor:
    def __init__(self, q, m, t, chosen_indices):
        self.__q = q
        self.__GF = galois.GF(q)
        self.__number_of_rows = m - t + 1
        self.__number_of_cols = m + 1
        self.__chosen_points = self.__translate_chosen_indices_to_points(chosen_indices)

    def __translate_chosen_indices_to_points(self, chosen_points_indices):
        chosen_points = np.zeros((self.__number_of_cols, len(chosen_points_indices)), dtype=int)
        for point_i, chosen_point_index in enumerate(chosen_points_indices):
            n = 1
            current_sum = 1
            while current_sum < chosen_point_index + 1:
                n += 1
                current_sum += (self.__q ** (n - 1))

            left = (chosen_point_index + 1) - int((self.__q ** (n - 1) - 1) // (self.__q - 1)) - 1
            point = np.zeros(self.__number_of_cols, dtype=int)
            for i in range(n - 1):
                point[i] = left % self.__q
                left = math.floor(left / self.__q)
            point[n - 1] = 1
            point = np.flip(point)
            chosen_points[:, point_i] = point.T
        return self.__GF(chosen_points)

    def construct_induced_covering(self):
        prev_matrix = None
        multiplication_result = None
        for matrix in reduced_row_echelon_form_gray_code(self.__number_of_rows, self.__number_of_cols, self.__q):
            if multiplication_result is None:
                multiplication_result = self.__GF(matrix) @ self.__chosen_points
            else:
                different_indices = np.argwhere(matrix != prev_matrix)
                assert different_indices.shape[0] == 1
                row_index, col_index = tuple(different_indices[0])
                row_delta = (self.__GF(matrix[row_index, col_index]) - self.__GF(prev_matrix[row_index, col_index])) * self.__chosen_points[col_index, :]
                multiplication_result[row_index] += row_delta

            block = np.nonzero(np.max(multiplication_result.view(np.ndarray), axis=0) == 0)[0]
            yield block

            prev_matrix = matrix


from subprocess import Popen
import sys
from pathlib import Path

def launch_covering_generator(q: int, m: int, t: int, chosen_indices: list[int], port: int, num_workers=1, worker_id=0) -> Popen:
    return Popen([
        sys.executable, "-m", Path(__file__).parent.name,
        '--q', str(q), '--m', str(m), '--t', str(t),
        '--chosen_indices', *map(str, chosen_indices),
        '--port', str(port),
        '--num-workers', str(num_workers),
        '--worker-id', str(worker_id)
    ])
