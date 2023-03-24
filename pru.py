import netsquid as ns
import numpy as np

from netsquid.qubits.qubitapi import *
from netsquid.qubits.qformalism import *
from netsquid.qubits import operators as ops
from netsquid.qubits import ketstates
from netsquid.qubits import Stabilizer

from netsquid.qubits import set_qstate_formalism, QFormalism


set_qstate_formalism(QFormalism.KET)
q1, q2, q3 = create_qubits(num_qubits=3, system_name="Q")
print(ns.qubits.reduced_dm(q1))  
print(reduced_dm([q1, q2]))
print(ns.qubits.ketstates.b00)
#print(ns.qubits.qstate(q1))

print("")
q1, q2 = create_qubits(2)
print(ns.qubits.qrepr(2))
print("")
assign_qstate([q1, q2], np.diag([0.25, 0.25, 0.25, 0.25]))
print(q1.qstate.qrepr) 