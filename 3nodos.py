import netsquid as ns
import pandas as pd
import numpy as np
import pydynaa as py

from netsquid.qubits import operators as ops

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

def create_network(distance, att, depol):
	network = Network("3 nodes network")
	node_A = Node(name="A", qmemory = create_qprocessor("MemoriaA"))
	node_B = Node(name="B", qmemory = create_qprocessor("MemoriaB"))
	node_C = Node(name="C", qmemory = create_qprocessor("MemoriaC"))
	delay_model = PingPongDelayModel()
	channel_1 = QuantumChannel(name="qchannel[A to B]", length=distance, models={"fibre_depolarize_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
	channel_2 = QuantumChannel(name="qchannel[B to A]", length=distance, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=None)})
	channel_3 = QuantumChannel(name="qchannel[B to C]", length=distance, models={"fibre_depolarize_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
	channel_4 = QuantumChannel(name="qchannel[C to B]", length=distance, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=None)})
	conn = DirectConnection(name="conn[A|B]",	channel_AtoB=channel_1, channel_BtoA=channel_2)
	con = DirectConnection(name="conn[B|C]",	channel_AtoB=channel_3, channel_BtoA=channel_4)
	channel_1.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	channel_3.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	#network.add_connection(node_A, node_B, connection=conn)
	node_A.connect_to(remote_node=node_B, connection=conn, local_port_name="qubitIO", remote_port_name="qubitIO")
	node_B.connect_to(remote_node=node_C, connection=con, local_port_name="qubitBC", remote_port_name="qubitBC")
	network.add_nodes([node_A, node_B, node_C])
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
            #ns.qubits.apply_dda_noise(qubit, depol=prob, deph=0.5, ampl=0.1)
            #ns.qubits.qubitapi.apply_pauli_noise(qubit, (0, prob, 0, (1-prob)))
            #ns.qubits.dephase(qubit, prob=prob)
                      
class ProtocolA(NodeProtocol):
	def __init__(self, node, qubit=None):
		super().__init__(node)
		self.qubit = qubit
		
	def run(self):
		if self.qubit is not None:
			#self.node.qmemory.put(qubits=self.qubit, positions=0, replace=True)
			self.node.ports["qubitIO"].tx_output(self.qubit)
		while True:
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0] #|0>
			self.node.ports["qubitIO"].tx_output(qubit)
			tiempo = ns.sim_time(ns.SECOND) + 1 #Esperamos 1 segundo para mandar el siguiente qubit
			yield self.await_timer(tiempo)

class ProtocolB(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitIO"])
			message = self.node.ports["qubitIO"].rx_input()
			qubit = message.items[0]
			fid = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=True)
			global fidB
			global cont
			fidB = fidB + fid
			cont += 1
			self.node.ports["qubitBC"].tx_output(qubit)
						
class ProtocolC(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitBC"])
			message = self.node.ports["qubitBC"].rx_input()
			qubit = message.items[0]
			self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
			#if qubit != ([[1.+0.j], [0.+0.j]]):
				#self.node.qmemory.execute_instruction(instr.INSTR_Z)
			if self.node.qmemory.busy:
				yield self.await_program(self.node.qmemory)
			self.send_signal(Signals.SUCCESS)
			
def setup_datacollector(network, protocol):
	nodoC = network.get_node("C")
	
	def calc_fidelity(evexpr):
		qubitC, = nodoC.qmemory.peek([0]) #cogemos el qubit de la memoria del receptor
		global num
		global rep 
		fidelity = ns.qubits.fidelity(qubitC, ns.qubits.ketstates.s0, squared=True) #calculamos la fidelidad
		num += 1 #aumentamos el contador de fidelidades calculadas
		if num >= rep: #si ha calculado 2000 qubits paramos la simulación
			ns.sim_stop()
			print(f"{num}")
			num = 0
		return {"fidelity": fidelity} #devolvemos el resultado de la fidelidad
		
	dc = DataCollector(calc_fidelity, include_entity_name=False)
	dc.collect_on(py.EventExpression(source=protocol, event_type=Signals.SUCCESS.value)) #calculamos la fidelidad cuando recibamos la señal del receptor
	return dc

num = 0
rep = 0
fidB = 0	
cont = 0
def run_simulation(distance, att, depol):
	ns.sim_reset()
	network = create_network(distance, att, depol)
	a_protocol = ProtocolA(network.get_node("A"), qubit=None)
	b_protocol = ProtocolB(network.get_node("B"))
	c_protocol = ProtocolC(network.get_node("C"))
	dc = setup_datacollector(network, c_protocol)
	a_protocol.start()
	b_protocol.start()
	c_protocol.start()
	ns.sim_run(end_time=10e9, magnitude=1e9)
	return dc.dataframe
	
def create_plot(num_iters):
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	global fidB
	global cont
	global rep
	rep = num_iters
	att = 0.15
	#depolar = [i for i in range(500, 1100, 100)]
	depolar = [0.01, 0.05, 0.025, 0.05, 0.07]
	dist = [2, 5, 10, 15, 20, 25, 30]
	for depol in depolar:
		fid = []
		data = pd.DataFrame()
		for distance in dist:
			data[distance] = run_simulation(distance=distance, att=att, depol=depol)['fidelity']
			print(f"cont: {cont}")
			fid.append(fidB/cont)
			print(f"Fidelidad B: {fidB/cont}")
			cont = 0
			fidB = 0
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		data.plot(y='fidelity', yerr='sem', label=f"{depol} C", ax=ax)
		plt.plot(dist, fid, 'o-', label=f"{depol} B")
		print(f"Depolarización: {depol}")
		print(f"{data}")
		print("")
	plt.xlabel("Distancia entre nodos (km)")
	plt.ylabel("Fidelidad")
	plt.title("Conexión directa")
	plt.legend(title='ratio de despolarización')
	plt.grid()
	plt.show()
	
if __name__ == "__main__":
	create_plot(2000)
	
	