import netsquid as ns
import pandas as pd
import numpy as np
import pydynaa as py

from netsquid.protocols.protocol import *
from netsquid.components.component import *
from netsquid.util.simstats import *
from netsquid.components.models.delaymodels import *	
from netsquid.components.models import QuantumErrorModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.qsource import *
from netsquid.components.qchannel import *
from netsquid.qubits.state_sampler import *
from netsquid.components.models.qerrormodels import *
from netsquid.protocols import NodeProtocol
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.qprocessor import PhysicalInstruction
from netsquid.components.qprocessor import QuantumProcessor
from netsquid.nodes.connections import *
from netsquid.protocols.nodeprotocols import *
from netsquid.util.datacollector import *

from netsquid.components.instructions import *

from netsquid.nodes import *

from netsquid.components import *

def create_qprocessor(name):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has two memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    name : str
        Name of the quantum processor.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    noise_rate = 200
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration,
                            q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration,
                            q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False, mem_noise_models=[mem_noise_model] * 2, phys_instructions=physical_instructions)
    return qproc
    
def setup_network(distance, source_frequency, att):
    network = Network("Red")
    node_Alice = Node("node_Alice", qmemory=create_qprocessor("processor_Alice"))
    node_Bob = Node(name="node_Bob")
    node_Bob = PongProtocol(node_Bob, observable=ns.X)
    network.add_nodes[node_Alice, node_Bob]
    channel = QuantumChannel(name="qchannel[A to B]", length=distance, models={"delay_model": delay_model, "quantum_loss_model": FibreLossModel(p_loss_init=0.09, p_loss_length=0.2, rng=None)})
    connection = DirectConnection(name="conexion",	channel_AtoB=channel, channel_BtoA=channel)
    node_Alice.connect_to(remote_node=node_Bob, connection=connection, local_port_name="port_Alice", remote_port_name="port_Bob")
    qsource = QSource(f"qsource_{name}", StateSampler([ns.qubits.ketstates.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e9 / source_frequency), status=SourceStatus.INTERNAL)
    self.add_subcomponent(qsource, name="qsource")
    qsource.ports["port_Alice"].connect(channel.ports["send"])
    return network
  
class PongProtocol(NodeProtocol):
    def __init__(self, node, observable, qubit=None):
        super().__init__(node)
        self.observable = observable
        self.qubit = qubit
        qmemory=create_qprocessor("processor_Bob")
        
        # Define matching pair of strings for pretty printing of basis states:
        self.basis = ["|+>", "|->"]

    def run(self):
        while True:
            # Wait (yield) until input has arrived on our port:
            yield self.await_port_input(self.node.ports["port_Bob"]) 
            # Receive (RX) qubit on the port's input:
            message = self.node.ports["port_Bob"].rx_input()
            qubit = message.items[0]
            meas, prob = ns.qubits.measure(qubit, observable=self.observable)
            print(f"{ns.sim_time():5.1f}: {self.node.name} measured " f"{self.basis[meas]} with probability {prob:.2f}")
            	  
def run_simulation(distance=20, num_iters=100, att=0.2):
    """Run the simulation experiment and return the collected data.

    Parameters
    ----------
    num_nodes : int, optional
        Number nodes in the repeater chain network. At least 3. Default 4.
    node_distance : float, optional
        Distance between nodes, larger than 0. Default 20 [km].
    num_iters : int, optional
        Number of simulation runs. Default 100.
    p_loss_length : float, optional
    	  Atenuation per km. Default 0.2.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with recorded fidelity data.

    """
    ns.sim_reset()
    print("run sim")
    #print("atenuación: "f"{att}")
    est_runtime = 1.5 * distance * 5e3
    network = setup_network(distance=distance, source_frequency=1e9 / est_runtime, att=att)
    PongProtocol.start()
    run_stats = ns.sim_run(est_runtime * num_iters)
    print(run_stats)
    
run_simulation(2,10,0.2)
    



