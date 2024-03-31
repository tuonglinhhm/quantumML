"""
Qubit Rotation Optimization, adapted from https://pennylane.ai/qml/demos/tutorial_qubit_rotation
"""

# import dependencies
import torchquantum as tq
import torch
from torchquantum.measurement import expval_joint_analytical
import argparse


class OptimizationModel(torch.nn.Module):
    """
    Circuit with rx and ry gate
    """

    def __init__(self):
        super().__init__()
        self.rx0 = tq.RX(has_params=True, trainable=True, init_params=0.011)
        self.ry0 = tq.RY(has_params=True, trainable=True, init_params=0.012)

    def forward(self):
        # create a quantum device to run the gates
        qdev = tq.QuantumDevice(n_wires=1)

        # add some trainable gates (need to instantiate ahead of time)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=0)

        # return the analytic expval from Z
        return expval_joint_analytical(qdev, "Z")


# train function to get expval as low as possible (ideally -1)
def train(model, device, optimizer):
    outputs = model()
    loss = outputs
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# main function to run the optimization
def main(n_epochs):
    seed = 0
    torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = OptimizationModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(1, n_epochs + 1):
        # train
        loss = train(model, device, optimizer)
        output = (model.rx0.params[0].item(), model.ry0.params[0].item())

        print(f"Epoch {epoch}: {output}")

        if epoch % 10 == 0:
            print(f"Loss after step {epoch}: {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Qubit Rotation",
        description="Specify Parameters for Qubit Rotation Optimization Example",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )
    args = parser.parse_args()
    main(args.epochs)
