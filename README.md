
## Installation [Original library from Torchquantum]
```bash
git clone https://github.com/mit-han-lab/torchquantum.git
cd torchquantum
pip install --editable .
```

## Basic Usage

```python
import torchquantum as tq
import torchquantum.functional as tqf

qdev = tq.QuantumDevice(n_wires=2, bsz=5, device="cpu", record_op=True) # use device='cuda' for GPU

# use qdev.op
qdev.h(wires=0)
qdev.cnot(wires=[0, 1])

# use tqf
tqf.h(qdev, wires=1)
tqf.x(qdev, wires=1)

# use tq.Operator
op = tq.RX(has_params=True, trainable=True, init_params=0.5)
op(qdev, wires=0)

# print the current state (dynamic computation graph supported)
print(qdev)

# obtain the qasm string
from torchquantum.plugin import op_history2qasm
print(op_history2qasm(qdev.n_wires, qdev.op_history))

# measure the state on z basis
print(tq.measure(qdev, n_shots=1024))

# obtain the expval on a observable by stochastic sampling (doable on simulator and real quantum hardware)
from torchquantum.measurement import expval_joint_sampling
expval_sampling = expval_joint_sampling(qdev, 'ZX', n_shots=1024)
print(expval_sampling)

# obtain the expval on a observable by analytical computation (only doable on classical simulator)
from torchquantum.measurement import expval_joint_analytical
expval = expval_joint_analytical(qdev, 'ZX')
print(expval)

# obtain gradients of expval w.r.t. trainable parameters
expval[0].backward()
print(op.params.grad)


# Apply gates to qdev with tq.QuantumModule
ops = [
    {'name': 'hadamard', 'wires': 0}, 
    {'name': 'cnot', 'wires': [0, 1]},
    {'name': 'rx', 'wires': 0, 'params': 0.5, 'trainable': True},
    {'name': 'u3', 'wires': 0, 'params': [0.1, 0.2, 0.3], 'trainable': True},
    {'name': 'h', 'wires': 1, 'inverse': True}
]

qmodule = tq.QuantumModule.from_op_history(ops)
qmodule(qdev)
```


<!--
## Basic Usage 2

```python
import torchquantum as tq
import torchquantum.functional as tqf

x = tq.QuantumDevice(n_wires=2)

tqf.hadamard(x, wires=0)
tqf.x(x, wires=1)
tqf.cnot(x, wires=[0, 1])

# print the current state (dynamic computation graph supported)
print(x.states)

# obtain the classical bitstring distribution
print(tq.measure(x, n_shots=2048))
```
 -->


## Typical examples


* [QNN for MNIST](examples/mnist)
* [Quantum Convolution (Quanvolution)](examples/quanvolution)
* [Quantum Regression](examples/regression)
* [VQE](examples/vqe)
* [QAOA (Quantum Approximate Optimization Algorithm)](examples/qaoa).


## Usage

Construct parameterized quantum circuit models as simple as constructing a normal pytorch model.
```python
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

class QFCModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.n_wires = 4
    self.measure = tq.MeasureAll(tq.PauliZ)

    self.encoder_gates = [tqf.rx] * 4 + [tqf.ry] * 4 + \
                         [tqf.rz] * 4 + [tqf.rx] * 4
    self.rx0 = tq.RX(has_params=True, trainable=True)
    self.ry0 = tq.RY(has_params=True, trainable=True)
    self.rz0 = tq.RZ(has_params=True, trainable=True)
    self.crx0 = tq.CRX(has_params=True, trainable=True)

  def forward(self, x):
    bsz = x.shape[0]
    # down-sample the image
    x = F.avg_pool2d(x, 6).view(bsz, 16)

    # create a quantum device to run the gates
    qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)

    # encode the classical image to quantum domain
    for k, gate in enumerate(self.encoder_gates):
      gate(qdev, wires=k % self.n_wires, params=x[:, k])

    # add some trainable gates (need to instantiate ahead of time)
    self.rx0(qdev, wires=0)
    self.ry0(qdev, wires=1)
    self.rz0(qdev, wires=3)
    self.crx0(qdev, wires=[0, 2])

    # add some more non-parameterized gates (add on-the-fly)
    qdev.h(wires=3)
    qdev.sx(wires=2)
    qdev.cnot(wires=[3, 0])
    qdev.qubitunitary(wires=[1, 2], params=[[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 0, 1j],
                                            [0, 0, -1j, 0]])

    # perform measurement to get expectations (back to classical domain)
    x = self.measure(qdev).reshape(bsz, 2, 2)

    # classification
    x = x.sum(-1).squeeze()
    x = F.log_softmax(x, dim=1)

    return x

```

## VQE Example

Train a quantum circuit to perform VQE task.
Quito quantum computer as in [simple_vqe.py](./examples/simple_vqe/simple_vqe.py)
script:
```python
cd examples/vqe
python vqe.py
```

## MNIST Example

Train a quantum circuit to perform MNIST classification task and deploy on the real IBM
Quito quantum computer as in [mnist_example.py](./examples/simple_mnist/mnist_example_no_binding.py)
script:
```python
cd examples/mnist
python mnist.py
```

## Files

| File      | Description |
| ----------- | ----------- |
| devices.py      | QuantumDevice class which stores the statevector |
| encoding.py   | Encoding layers to encode classical values to quantum domain |
| functional.py   | Quantum gate functions |
| operators.py   | Quantum gate classes |
| layers.py   | Layer templates such as RandomLayer |
| measure.py   | Measurement of quantum states to get classical values |
| graph.py   | Quantum gate graph used in static mode |
| super_layer.py   | Layer templates for SuperCircuits |
| plugins/qiskit*   | Convertors and processors for easy deployment on IBMQ |
| examples/| More examples for training QML and VQE models |

## Coding Style

torchquantum uses pre-commit hooks to ensure Python style consistency and prevent common mistakes in its codebase.

To enable it pre-commit hooks please reproduce:
```bash
pip install pre-commit
pre-commit install
```


[comment]: <> (## More Examples)

[comment]: <> (The `examples/` folder contains more examples to train the QML and VQE)

[comment]: <> (models. Example usage for a QML circuit:)

[comment]: <> (```python)

[comment]: <> (# train the circuit with 36 params in the U3+CU3 space)

[comment]: <> (python examples/train.py examples/configs/mnist/four0123/train/baseline/u3cu3_s0/rand/param36.yml)

[comment]: <> (# evaluate the circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/tq/all.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (# evaluate the circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/mnist/four0123/eval/x2/real/opt2/300.yml --run-dir=runs/mnist.four0123.train.baseline.u3cu3_s0.rand.param36)

[comment]: <> (```)

[comment]: <> (Example usage for a VQE circuit:)

[comment]: <> (```python)

[comment]: <> (# Train the VQE circuit for h2)

[comment]: <> (python examples/train.py examples/configs/vqe/h2/train/baseline/u3cu3_s0/human/param12.yml)

[comment]: <> (# evaluate the VQE circuit with torchquantum)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/tq/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (# evaluate the VQE circuit with real IBMQ-Yorktown quantum computer)

[comment]: <> (python examples/eval.py examples/configs/vqe/h2/eval/x2/real/opt2/all.yml --run-dir=runs/vqe.h2.train.baseline.u3cu3_s0.human.param12/)

[comment]: <> (```)



## Dependencies

- 3.9 >= Python >= 3.7 (Python 3.10 may have the `concurrent` package issue for Qiskit)
- PyTorch >= 1.8.0
- configargparse >= 0.14
- GPU model training requires NVIDIA GPUs
