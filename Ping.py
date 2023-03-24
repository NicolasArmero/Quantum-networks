import netsquid as ns

qubits = ns.qubits.create_qubits(1)
qubits
qubit = qubits[0]
# To check the state is |0> we check its density matrix using reduced_dm():
ns.qubits.reduced_dm(qubit)

ns.qubits.operate(qubit, ns.X)
ns.qubits.reduced_dm(qubit)

from netsquid.nodes import Node
node_ping = Node(name="Ping")
node_pong = Node(name="Pong")

from netsquid.components.models import DelayModel
from netsquid.components.models import QuantumErrorModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.protocols.protocol import *

class PingPongDelayModel(DelayModel):
    def __init__(self, speed_of_light_fraction=0.5, standard_deviation=0.05):
        super().__init__()
        # (the speed of light is about 300,000 km/s)
        self.properties["speed"] = speed_of_light_fraction * 3e5
        self.properties["std"] = standard_deviation
        self.required_properties = ['length']  # in km

    def generate_delay(self, **kwargs):
        avg_speed = self.properties["speed"]
        std = self.properties["std"]
        # The 'rng' property contains a random number generator
        # We can use that to generate a random speed
        speed = self.properties["rng"].normal(avg_speed, avg_speed * std)
        delay = 1e9 * kwargs['length'] / speed  # in nanoseconds
        return delay
         
from netsquid.components import QuantumChannel

distance = 50  # default unit of length in channels is km
delay_model = PingPongDelayModel()
channel_1 = QuantumChannel(name="qchannel[ping to pong]", length=distance, models={"delay_model": delay_model, "quantum_loss_model": FibreLossModel(p_loss_init=0.09, p_loss_length=0.2, rng=None)})
channel_2 = QuantumChannel(name="qchannel[pong to ping]", length=distance, models={"delay_model": delay_model, "quantum_loss_model": FibreLossModel(p_loss_init=0.09, p_loss_length=0.2, rng=None)})
                           
from netsquid.nodes import DirectConnection

connection = DirectConnection(name="conn[ping|pong]",	channel_AtoB=channel_1, channel_BtoA=channel_2)
node_ping.connect_to(remote_node=node_pong, connection=connection, local_port_name="qubitIO", remote_port_name="qubitIO")
                     

from netsquid.protocols import NodeProtocol

class PingProtocol(NodeProtocol):
    def __init__(self, node, observable, qubit=None):
        super().__init__(node)
        self.observable = observable
        self.qubit = qubit
        # Define matching pair of strings for pretty printing of basis states:
        self.basis = ["|0>", "|1>"] if observable == ns.Z else ["|+>", "|->"]

    def run(self):
        if self.qubit is not None:
            # Send (TX) qubit to the other node via port's output:
            self.node.ports["qubitIO"].tx_output(self.qubit)
        while True:
            # Wait (yield) until input has arrived on our port:
            #transito = 50*2/3e5
            tiempo = ns.sim_time() + 1e9/2
            #print(f"Tiempo: {tiempo}")
            #print(f"Sim: {ns.sim_time()}")
            yield self.await_timer(tiempo)
            # Receive (RX) qubit on the port's input:
            #message = self.node.ports["qubitIO"].rx_input()
            qubits = ns.qubits.create_qubits(1)
            qubit = qubits[0]
            global numPing
            meas, prob = ns.qubits.measure(qubit, observable=self.observable)
            numPing += 1 
            #print(f"{self.node.name} measured "f"{self.basis[meas]} with probability {prob:.2f}")
            # Send (TX) qubit to the other node via connection:
            self.node.ports["qubitIO"].tx_output(qubit)
            
class PongProtocol(NodeProtocol):
    def __init__(self, node, observable, qubit=None):
        super().__init__(node)
        self.observable = observable
        self.qubit = qubit
        # Define matching pair of strings for pretty printing of basis states:
        self.basis = ["|0>", "|1>"] if observable == ns.Z else ["|+>", "|->"]

    def run(self):
        if self.qubit is not None:
            # Send (TX) qubit to the other node via port's output:
            self.node.ports["qubitIO"].tx_output(self.qubit)
        while True:
            # Wait (yield) until input has arrived on our port:
            yield self.await_port_input(self.node.ports["qubitIO"])
            # Receive (RX) qubit on the port's input:
            message = self.node.ports["qubitIO"].rx_input()
            qubit = message.items[0]
            meas, prob = ns.qubits.measure(qubit, observable=self.observable)
            print(f"{self.node.name} measured "f"{self.basis[meas]} with probability {prob:.2f}")
            global numPong
            numPong += 1
            if numPong >= 100:
            	#print(f"num: {num}")
            	ns.sim_stop()
            # Send (TX) qubit to the other node via connection:
            #self.node.ports["qubitIO"].tx_output(qubit)
           
qubits = ns.qubits.create_qubits(1)
ping_protocol = PingProtocol(node_ping, observable=ns.X, qubit=qubits[0])
pong_protocol = PongProtocol(node_pong, observable=ns.X)
			
ping_protocol.start()
pong_protocol.start()
est_runtime = 1.5 * 50 * 5e3
numPing = 0
numPong = 0
run_stats = ns.sim_run()
print(f"Pings enviados: {numPing}")
print(f"Pings recibidos: {numPong}")
print(run_stats)