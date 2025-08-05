# Reference: https://github.com/YuvShap/finite-geometry-coverings-construction
import numpy as np


# Translate to base q (LSB at lowest index)
def to_base_q(q, length, index):
    assert index < q ** length
    q_ary_representation = []
    for _ in range(length):
        q_ary_representation.append(index % q)
        index //= q
    return q_ary_representation


# Gray code that starts with 100...0 and ends with 00...0,
# based on: https://en.wikipedia.org/wiki/Gray_code#n-ary_Gray_code
def q_ary_vector_gray_code(q, length, index):
    index = (index + 1) % (q ** length)
    q_ary_representation = to_base_q(q, length, index)
    result = []
    temp = 0
    for symbol in reversed(q_ary_representation):
        gray_symbol = (symbol + temp) % q
        temp += ((q - gray_symbol) % q)
        result.append(gray_symbol)
    return list(reversed(result))


# Based on Theorem 4.1 in "On the q-analog of the revolving door algorithm"
# by Braun and Mallmann: https://ajc.maths.uq.edu.au/pdf/71/ajc_v71_p261.pdf
def reduced_column_echelon_form_gray_code(n, k, q):
    if n == k:
        yield np.eye(n, dtype=int)
        return
    if k == 0:
        yield np.zeros((n, 0), dtype=int)
        return

    new_column = np.zeros(n, dtype=int)
    new_column[0] = 1
    new_column = new_column.reshape(n, 1)
    for matrix in reduced_column_echelon_form_gray_code(n - 1, k - 1, q):
        temp = np.vstack((np.zeros((1, k - 1), dtype=int), matrix))
        new_matrix = np.hstack((new_column, temp))
        yield new_matrix

    k_gray_code = [q_ary_vector_gray_code(q, k, i) for i in range(0, q ** k)]
    index = 0
    examined_matrices = []
    for matrix in reduced_column_echelon_form_gray_code(n - 1, k, q):
        if index % 2 == 0:
            for word in k_gray_code[:-1]:
                new_row = np.asarray(word, dtype=int).reshape(1, k)
                new_matrix = np.vstack((new_row, matrix))
                yield new_matrix
        else:
            for word in reversed(k_gray_code[:-1]):
                new_row = np.asarray(word, dtype=int).reshape(1, k)
                new_matrix = np.vstack((new_row, matrix))
                yield new_matrix

        examined_matrices.append(matrix)
        index += 1

    word = k_gray_code[-1]
    new_row = np.asarray(word, dtype=int).reshape(1, k)
    for matrix in reversed(examined_matrices):
        new_matrix = np.vstack((new_row, matrix))
        yield new_matrix


def reduced_row_echelon_form_gray_code(number_of_rows, number_of_cols, q):
    for matrix in reduced_column_echelon_form_gray_code(number_of_cols, number_of_rows, q):
        yield matrix[::-1, ::-1].T
