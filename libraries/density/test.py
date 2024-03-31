import torch

from torchquantum.density import density_mat
from torchquantum.density import density_func

if __name__ == "__main__":
    mat = density_func.mat_dict["hadamard"]

    Xgatemat = density_func.mat_dict["paulix"]
    print(mat)
    D = density_mat.DensityMatrix(2, 2)

    rho = torch.zeros(2 ** 4,)
    rho = torch.reshape(rho, [4, 4])
    rho[0][0] = 1 / 2
    rho[0][3] = 1 / 2
    rho[3][0] = 1 / 2
    rho[3][3] = 1 / 2
    rho = torch.reshape(rho, [2, 2, 2, 2])
    D.update_matrix(rho)
    D.print_2d(0)
    newD = density_func.apply_unitary_density_bmm(D._matrix, Xgatemat, [1])

    D.update_matrix(newD)

    D.print_2d(0)

