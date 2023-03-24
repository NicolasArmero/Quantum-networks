import netsquid as ns
import pandas as pd
import numpy as np
import pydynaa as py

from netsquid.components import QuantumChannel
from netsquid.nodes import DirectConnection
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.protocols import NodeProtocol
from netsquid.nodes import Node, Network
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.util.datacollector import DataCollector
from netsquid.components.models import QuantumErrorModel, DelayModel
import netsquid.components.instructions as instr
from netsquid.protocols.protocol import *

def create_network(distance, att):
	network = Network("Ping network")
	node_ping = Node(name="Ping", qmemory = create_qprocessor("MemoriaPing"))
	node_pong = Node(name="Pong", qmemory = create_qprocessor("MemoriaPong"))
	delay_model = PingPongDelayModel()
	channel_1 = QuantumChannel(name="qchannel[ping to pong]", length=distance, models={"quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
	channel_2 = QuantumChannel(name="qchannel[pong to ping]", length=distance, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=None)})
	conn = DirectConnection(name="conn[ping|pong]",	channel_AtoB=channel_1, channel_BtoA=channel_2)
	#channel_1.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	#network.add_connection(node_ping, node_pong, connection=conn)
	node_ping.connect_to(remote_node=node_pong, connection=conn, local_port_name="qubitIO", remote_port_name="qubitIO")
	network.add_nodes([node_ping, node_pong])
	return network
	
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
	
def create_qprocessor(name):
	noise_rate = 1e7
	gate_duration = 1
	gate_noise_model = DephaseNoiseModel(noise_rate)
	mem_noise_model = DepolarNoiseModel(noise_rate)
	physical_instructions = [
		PhysicalInstruction(instr.INSTR_X, duration=gate_duration, q_noise_model=gate_noise_model),
		PhysicalInstruction(instr.INSTR_Z, duration=gate_duration, q_noise_model=gate_noise_model),
	]
	qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False, mem_noise_models=[mem_noise_model] * 2, phys_instructions=physical_instructions)
	return qproc

class FibreDepolarizeModel(QuantumErrorModel):
    def __init__(self, p_depol_init=0, p_depol_length=0.025):
        super().__init__()
        self.properties['p_depol_init'] = p_depol_init
        self.properties['p_depol_length'] = p_depol_length
        self.required_properties = ['length']

    def error_operation(self, qubits, delta_time=0, **kwargs):
        """Uses the length property to calculate a depolarization probability, and applies it to the qubits.

        Parameters
        ----------
        qubits : tuple of :obj:`~netsquid.qubits.qubit.Qubit`
            Qubits to apply noise to.
        delta_time : float, optional
            Time qubits have spent on a component [ns]. Not used.

        """
        for qubit in qubits:
            prob = 1 - (1 - self.properties['p_depol_init']) * np.power(10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
            ns.qubits.depolarize(qubit, prob=prob)
                      
class PingProtocol(NodeProtocol):
	def __init__(self, node, qubit=None):
		super().__init__(node)
		self.qubit = qubit
		
	def run(self):
		if self.qubit is not None:
			#self.node.qmemory.put(qubits=self.qubit, positions=0, replace=True)
			self.node.ports["qubitIO"].tx_output(self.qubit)
		while True:
			tiempo = ns.sim_time(ns.SECOND) + 0.01 #Esperamos 1 us para mandar el siguiente qubit
			yield self.await_timer(tiempo)
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0]
			global ping
			ping += 1
			if ping >= 2000:
				ns.sim_stop()
			#self.node.qmemory.put(qubits=q1, positions=0, replace=True)
			#fid = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=False)
			#print(f"Fid: {fid}")
			self.node.ports["qubitIO"].tx_output(qubit)
			
class PongProtocol(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitIO"])
			message = self.node.ports["qubitIO"].rx_input()
			qubit = message.items[0]
			#self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
			#self.node.qmemory.execute_instruction(instr.INSTR_Z)
			#if self.node.qmemory.busy:
				#yield self.await_program(self.node.qmemory)
			global pong
			pong += 1
			#if pong >= 2000:
				#ns.sim_stop()
			#self.send_signal(Signals.SUCCESS)
			
def setup_datacollector(network, protocol):
	nodoA = network.get_node("Ping")
	nodoB = network.get_node("Pong")
	
	def calc_fidelity(evexpr):
		#qubitA, = nodoA.qmemory.peek([0])
		qubitB, = nodoB.qmemory.peek([0])
		global num
		fidelity = ns.qubits.fidelity(qubitB, ns.qubits.ketstates.s0, squared=True)
		#if qubitA is not None:
			#fidelity = ns.qubits.fidelity([qubitA, qubitB], ns.qubits.ketstates.b00, squared=False)
		num += 1
		#if num >= 2000:
			#ns.sim_stop()
			#num = 0
		return {"fidelity": fidelity}
		
	dc = DataCollector(calc_fidelity, include_entity_name=False)
	dc.collect_on(py.EventExpression(source=protocol, event_type=Signals.SUCCESS.value))
	return dc

num = 0
ping = 0
pong = 0
def run_simulation(distance, att):
	ns.sim_reset()
	network = create_network(distance, att)
	qubits = ns.qubits.create_qubits(1)
	ping_protocol = PingProtocol(network.get_node("Ping"), qubit=None)
	pong_protocol = PongProtocol(network.get_node("Pong"))
	#dc = setup_datacollector(network, pong_protocol)
	ping_protocol.start()
	pong_protocol.start()
	ns.sim_run(end_time=300, magnitude=1e9)
	#return dc.dataframe
	
def create_plot():
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	dist = [2, 5, 10, 15, 20, 25, 30, 40, 50]
	for att in [0.2, 0.25, 0.3, 0.35, 0.4]:
		data = pd.DataFrame()
		porc = []
		for distance in dist:
			run_simulation(distance=distance, att=att)
			global ping
			global pong
			porcentaje = (pong/ping)*100 #número de qubits que han llegado entre los que se han enviado %
			print(f"Ping: {ping} "f"Pong: {pong} "f"Porcentaje: {porcentaje}")
			ping = 0
			pong = 0
			#print(f"Porcentaje con atenuación {att} "f"a distancia {distance} "f"= {porcentaje}")
			porc.append(porcentaje)
		plt.plot(dist, porc, 'o-', label=f"{att} dB/km")
	plt.xlabel("Distancia (km)")
	plt.ylabel("Porcentaje")
	plt.title("Conexión directa")
	plt.legend(title='Atenuación')
	plt.grid()
	plt.show()
	
if __name__ == "__main__":
	create_plot()
	
	