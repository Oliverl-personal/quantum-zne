import pennylane as qml
from pennylane import math as math
from pennylane import numpy as np
from pennylane.queuing import QueuingManager
from functools import partial

from qiskit.providers.fake_provider import FakeVigo
from qiskit_aer import QasmSimulator
from qiskit_aer.noise import NoiseModel

@qml.qfunc_transform
def unitary_folding(tape, scale_factor): 
  for op in tape.operations:
    qml.apply(op)
  
  num_folds = math.round((scale_factor - 1.0) / 2.0)
  
  for _ in range(int(num_folds)):
    for op in tape.operations[::-1]:
      op.adjoint()
    for op in tape.operations: 
      qml.apply(op)
  
  for m in tape.measurements:
    qml.apply(m)

def fit_zne(scale_factors , energies):
  scale_factors = math.stack(scale_factors) 
  unwrapped_energies = math.stack(energies).ravel()

  N = len(energies)

  sum_scales = math.sum(scale_factors)
  sum_energies = math.sum(unwrapped_energies)

  numerator = N * math.sum(
    math.multiply(scale_factors , unwrapped_energies)
    ) - sum_scales * sum_energies
  
  denominator = N * math.sum(scale_factors ** 2) - sum_scales ** 2
  
  slope = numerator / denominator
  
  intercept = (sum_energies - slope * sum_scales) / N
  
  return intercept

@qml.batch_transform
def zne(tape, mitigation_transform, scale_factors): 
  with QueuingManager.stop_recording():
    tapes = [mitigation_transform.tape_fn(tape, scale) for scale in scale_factors]
  
  processing_fn = partial(fit_zne , scale_factors)
  
  return tapes , processing_fn

H = qml.Hamiltonian( 
  coeffs=[1.0, 2.0, 3.0],
  observables=[
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliZ(1) @ qml.PauliZ(2),
    qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2)
  ] 
)



if __name__ == "__main__":
  device = QasmSimulator.from_backend(FakeVigo())
  noise_model = NoiseModel.from_backend(device) 
  noisy_dev = qml.device(
    "qiskit.aer", backend='qasm_simulator', wires=3, shots=10000, noise_model=noise_model 
  )
  noisy_dev.set_transpile_args(**{"optimization_level" : 0})

  @zne(unitary_folding , [1.0, 3.0, 5.0, 7.0, 9.0])
  @qml.qnode(noisy_dev , diff_method='parameter-shift') 
  def circuit(params):
    qml.RX(params[0], wires=0) 
    qml.CNOT(wires=[0, 1])
    qml.RY(params[1], wires=1) 
    qml.CNOT(wires=[1, 2])
    qml.RZ(params[2], wires=2) 
    qml.CNOT(wires=[2, 0])
    return qml.expval(H)

  params = np.array([0.5, 0.1, -0.2], requires_grad=True)
  print(circuit(params))

  print(qml.grad(circuit)(params))

