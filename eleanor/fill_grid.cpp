/*
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

void fill_grid (py::array_t<double> grid_array, py::array_t<bool> mask_array, py::array_t<int> count_array) {
  auto grid = grid_array.mutable_unchecked<2>();
  auto mask = mask_array.unchecked<2>();
  auto count = count_array.mutable_unchecked<2>();

  for (ssize_t i = 0; i < grid.shape(0); ++i) {
    ssize_t left_ind = -1, right_ind = -1;
    double left = 0.0, right = 0.0;
    for (ssize_t j = 0; j < grid.shape(1); ++j) {
      if (mask(i, j)) {
        left = grid(i, j);
        left_ind = j;
      } else {
        if (j > right_ind) {
          for (ssize_t k = j+1; k < grid.shape(1); ++k) {
            if (mask(i, k)) {
              right = grid(i, k);
              right_ind = k;
              break;
            }
          }
        }
        if (left_ind < 0 && right_ind < 0) {
          // Uh oh! None of the points in this row are ok
          count(i, j) = 0;
        } else if (left_ind < 0) {
          grid(i, j) = right;
          count(i, j) = 1;
        } else if (right_ind < j) {
          grid(i, j) = left;
          count(i, j) = 1;
        } else {
          grid(i, j) = ((right_ind - j) * left + (j - left_ind) * right) / (right_ind - left_ind);
          count(i, j) = 1;
        }
      }
    }
  }

  for (ssize_t i = 0; i < grid.shape(1); ++i) {
    ssize_t left_ind = -1, right_ind = -1;
    double left = 0.0, right = 0.0;
    for (ssize_t j = 0; j < grid.shape(0); ++j) {
      if (mask(j, i)) {
        left = grid(j, i);
        left_ind = j;
      } else {
        if (j > right_ind) {
          for (ssize_t k = j+1; k < grid.shape(0); ++k) {
            if (mask(k, i)) {
              right = grid(k, i);
              right_ind = k;
              break;
            }
          }
        }
        if (left_ind < 0 && right_ind < 0) {
          // Uh oh! None of the points in this row are ok
          count(j, i) -= 1;
        } else if (left_ind < 0) {
          grid(j, i) += right;
        } else if (right_ind < j) {
          grid(j, i) += left;
        } else {
          grid(j, i) += ((right_ind - j) * left + (j - left_ind) * right) / (right_ind - left_ind);
        }
        grid(j, i) /= count(j, i) + 1;
      }
    }
  }

}

PYBIND11_MODULE(fill_grid, m) {
  m.def("fill_grid", &fill_grid, py::arg().noconvert(), py::arg().noconvert(), py::arg().noconvert());
}