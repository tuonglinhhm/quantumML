import functools
import torch
import numpy as np

from typing import Callable, Union, Optional, List, Dict, TYPE_CHECKING
from ..macro import C_DTYPE, F_DTYPE, ABC, ABC_ARRAY, INV_SQRT2
from ..util.utils import pauli_eigs, diag
from torchpack.utils.logging import logger
from torchquantum.util import normalize_statevector

from .gate_wrapper import gate_wrapper, apply_unitary_einsum, apply_unitary_bmm

if TYPE_CHECKING:
    from torchquantum.device import QuantumDevice
else:
    QuantumDevice = None


def cry_matrix(params):
    """Compute unitary matrix for CRY gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    matrix = (
        torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=C_DTYPE,
            device=params.device,
        )
        .unsqueeze(0)
        .repeat(co.shape[0], 1, 1)
    )
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 2, 3] = -si[:, 0]
    matrix[:, 3, 2] = si[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    return matrix.squeeze(0)


def ry_matrix(params: torch.Tensor) -> torch.Tensor:
    """Compute unitary matrix for ry gate.

    Args:
        params: The rotation angle.

    Returns:
        The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)

    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)

    return torch.stack(
        [torch.cat([co, -si], dim=-1), torch.cat([si, co], dim=-1)], dim=-2
    ).squeeze(0)


def ryy_matrix(params):
    """Compute unitary matrix for RYY gate.

    Args:
        params (torch.Tensor): The rotation angle.

    Returns:
        torch.Tensor: The computed unitary matrix.

    """
    theta = params.type(C_DTYPE)
    co = torch.cos(theta / 2)
    jsi = 1j * torch.sin(theta / 2)

    matrix = (
        torch.tensor(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=C_DTYPE,
            device=params.device,
        )
        .unsqueeze(0)
        .repeat(co.shape[0], 1, 1)
    )

    matrix[:, 0, 0] = co[:, 0]
    matrix[:, 1, 1] = co[:, 0]
    matrix[:, 2, 2] = co[:, 0]
    matrix[:, 3, 3] = co[:, 0]

    matrix[:, 0, 3] = jsi[:, 0]
    matrix[:, 1, 2] = -jsi[:, 0]
    matrix[:, 2, 1] = -jsi[:, 0]
    matrix[:, 3, 0] = jsi[:, 0]

    return matrix.squeeze(0)


def ryy(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the ryy gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "ryy"
    mat = _ry_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


_ry_mat_dict = {
    "ry": ry_matrix,
    "ryy": ryy_matrix,
    "cry": cry_matrix,
}


def cry(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the cry gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "cry"
    mat = _ry_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


def ry(
    q_device,
    wires,
    params=None,
    n_wires=None,
    static=False,
    parent_graph=None,
    inverse=False,
    comp_method="bmm",
):
    """Perform the ry gate.

    Args:
        q_device (tq.QuantumDevice): The QuantumDevice.
        wires (Union[List[int], int]): Which qubit(s) to apply the gate.
        params (torch.Tensor, optional): Parameters (if any) of the gate.
            Default to None.
        n_wires (int, optional): Number of qubits the gate is applied to.
            Default to None.
        static (bool, optional): Whether use static mode computation.
            Default to False.
        parent_graph (tq.QuantumGraph, optional): Parent QuantumGraph of
            current operation. Default to None.
        inverse (bool, optional): Whether inverse the gate. Default to False.
        comp_method (bool, optional): Use 'bmm' or 'einsum' method to perform
        matrix vector multiplication. Default to 'bmm'.

    Returns:
        None.

    """
    name = "ry"
    mat = _ry_mat_dict[name]
    gate_wrapper(
        name=name,
        mat=mat,
        method=comp_method,
        q_device=q_device,
        wires=wires,
        params=params,
        n_wires=n_wires,
        static=static,
        parent_graph=parent_graph,
        inverse=inverse,
    )


yy = ryy
