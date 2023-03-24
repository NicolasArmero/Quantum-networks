import pandas as pd
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.nodes import Node, Connection, Network, DirectConnection
from netsquid.protocols.protocol import Signals
from netsquid.protocols.nodeprotocols import NodeProtocol
from netsquid.components.qchannel import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.qerrormodels import DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.models.delaymodels import FibreDelayModel, FixedDelayModel
from netsquid.components.models.qerrormodels import FibreLossModel
from netsquid.components.models import QuantumErrorModel
from netsquid.util.datacollector import DataCollector
import netsquid as ns
import numpy as np
import pydynaa
import netsquid.qubits.ketstates as ks
import netsquid.qubits.qubitapi as qapi
import netsquid.components.instructions as instr

from netsquid.protocols.protocol import *
from netsquid.protocols.nodeprotocols import *
from netsquid.components.component import *


class EntanglingConnection(Connection):
	
	def __init__(self, length, source_frequency, name="EntanglingConnection"):
		super().__init__(name=name)
		qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e9 / source_frequency), status=SourceStatus.INTERNAL)
		self.add_subcomponent(qsource, name="qsource")
		qchannel_c2a = QuantumChannel("qchannel_C2A", length=length/2, models={"delay_model": FibreDelayModel()})
		qchannel_c2b = QuantumChannel("qchannel_C2B", length=length/2, models={"delay_model": FibreDelayModel()})
		# Add channels and forward quantum channel output to external port output:
		self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
		self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
		# Connect qsource output to quantum channel input:
		qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
		qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])

		
class ClassicalConnection(Connection):
	
	def __init__(self, length, name="ClassicalConnection"):
		super().__init__(name=name)
		self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length, models={"delay_model": FibreDelayModel()}), forward_input=[("A", "send")], forward_output=[("B", "recv")])


def create_processor(depolar_rate, dephase_rate):
    
    # We'll give both Alice and Bob the same kind of processor
    measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate, time_independent=True)
    physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=7, parallel=False),
        PhysicalInstruction(instr.INSTR_H, duration=4, parallel=False, topology=[0, 1]),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=False, topology=[0]),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=False, topology=[0]),
        PhysicalInstruction(instr.INSTR_S, duration=4, parallel=False, topology=[0]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=False, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[0], q_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[1]),
        PhysicalInstruction(instr.INSTR_MEASURE_BELL, duration=1)
    ]
    memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    processor = QuantumProcessor("quantum_processor", num_positions=2, memory_noise_models=[memory_noise_model] * 2, phys_instructions=physical_instructions)
    return processor
		
	
def example_network_setup(node_distance=4e-3, depolar_rate=1e7, dephase_rate=0.2):
	
	alice = Node("Alice", qmemory=create_processor(200, 0))
	bob = Node("Bob", qmemory=create_processor(200, 0))
	repetidor = Node("Rep", qmemory=create_processor(200, 0))
	creador = Node("Nodo0", qmemory=create_processor(200, 0))
	
	network = Network("Teleportation_network")
	network.add_nodes([alice, bob, repetidor, creador])
	
	conAB = ClassicalConnection(length=node_distance) #Enlace clásico de A a B
	conBA = ClassicalConnection(length=node_distance) #Enlace clásico de B a A
	conRB = ClassicalConnection(length=node_distance/2) #Enlace clásico de el rep a B
	conBR = ClassicalConnection(length=node_distance/2) #Enlace clásico de B al rep	
	conA0 = ClassicalConnection(length=0.01) #Enlace clásico de A a 0	
	network.add_connection(alice, creador, connection=conA0, label="classicalA0", port_name_node1="cout_0", port_name_node2="cin_alice")
	network.add_connection(alice, bob, connection=conAB, label="classical", port_name_node1="cout_bob", port_name_node2="cin_alice")
	network.add_connection(bob, alice, connection=conBA, label="classicalRA", port_name_node1="cout_alice", port_name_node2="cin_bob")
	network.add_connection(repetidor, bob, connection=conRB, label="classicalRB", port_name_node1="c2b", port_name_node2="cin_repB")
	network.add_connection(bob, repetidor, connection=conBR, label="classicalBR", port_name_node1="cout_repB", port_name_node2="cin_bob")
	
	channel_0A = QuantumChannel(name="qchannel[0 to A]", length=0.01, models={"fibre_delay_model": FibreDelayModel()})
	channel_A0 = QuantumChannel(name="qchannel[A to 0]", length=0.01, models={"fibre_delay_model": FibreDelayModel()})
	qcon0A = DirectConnection(name="conn[0|A]", channel_AtoB=channel_0A, channel_BtoA=channel_A0)
	port_0a, port_a0 = network.add_connection(creador, alice, connection=qcon0A, label="quantum", port_name_node1="qubitOUT", port_name_node2="qubitIN")
	alice.ports[port_a0].forward_input(alice.qmemory.ports['qin0'])
	
	source_frequency = 4e4 / node_distance
	q_conn = EntanglingConnection(name="qconn_A-R", length=node_distance/2, source_frequency=source_frequency)
	#for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
		#q_conn.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depolar_rate)
	port_ac, port_r2c = network.add_connection(alice, repetidor, connection=q_conn, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charlier")
	q_conn2 = EntanglingConnection(name="qconn_R-B", length=node_distance/2, source_frequency=source_frequency)
	#for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
		#q_conn2.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depolar_rate)
	port_r1c, port_bc = network.add_connection(repetidor, bob, connection=q_conn2, label="quantum", port_name_node1="qin_charliel", port_name_node2="qin_charlie")
 	
 	#Redirigimos la información de los puertos a la memoria u a otros nodos
	alice.ports[port_ac].forward_input(alice.qmemory.ports['qin1'])
	bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])
	repetidor.ports[port_r2c].forward_input(repetidor.qmemory.ports['qin1'])
	repetidor.ports[port_r1c].forward_input(repetidor.qmemory.ports['qin0'])
	return network
 	
 	
class SwapProtocol(NodeProtocol):
	
	def __init__(self, node, name):
		super().__init__(node, name)
		self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
		self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
		self._program = QuantumProgram(num_qubits=2)
		q1, q2 = self._program.get_qubit_indices(num_qubits=2)
		self._program.apply(instr.INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)
		
	def run(self):
		while True:
			yield (self.await_port_input(self._qmem_input_port_l) & self.await_port_input(self._qmem_input_port_r))
			#print("llegan a rep")
			# Perform Bell measurement
			yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
			m, = self._program.output["m"]
			#Manda resultado del Swap a Bob
			self.node.ports["c2b"].tx_output(Message(m))
	
	
class CorrectProtocol(NodeProtocol):
	
	def __init__(self, node, num_nodes):
		super().__init__(node, "CorrectProtocol")
		self.num_nodes = num_nodes
		self._x_corr = 0
		self._z_corr = 0
		self._program = SwapCorrectProgram()
		self._counter = 0	
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["cin_repB"]) #Espera hasta que recibe un mensaje
			#print("recibido")
			message = self.node.ports["cin_repB"].rx_input()
			if message is None or len(message.items) != 1:
				continue
			m = message.items[0]
			if m == ns.qubits.ketstates.BellIndex.B01 or m == ns.qubits.ketstates.BellIndex.B11:
				self._x_corr += 1
			if m == ns.qubits.ketstates.BellIndex.B10 or m == ns.qubits.ketstates.BellIndex.B11:
				self._z_corr += 1
			self._counter += 1
			if self._counter == self.num_nodes - 2: #Si es el nodo final y no un repetidor
				#print("entra")
				n = 1
				self.node.ports["cout_alice"].tx_output(n)
				if self._x_corr or self._z_corr:
					self._program.set_corrections(self._x_corr, self._z_corr)
					print("Entra aquí")
					yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1])
					print("Sale de aquí")
				#self.send_signal(Signals.SUCCESS)
				n = 1
				self.node.ports["cout_alice"].tx_output(n)
				#print("mensaje enviado")
				#self.node.ports["cout_alice"].tx_output(n)
				self._x_corr = 0
				self._z_corr = 0
				self._counter = 0
		

class SwapCorrectProgram(QuantumProgram):
	#Quantum processor program that applies all swap corrections.
	default_num_qubits = 1
	
	def set_corrections(self, x_corr, z_corr):
		self.x_corr = x_corr % 2
		self.z_corr = z_corr % 2

	def program(self):
		q1, = self.get_qubit_indices(1)
		if self.x_corr == 1:
			self.apply(instr.INSTR_X, q1)
			#print("Corr1 X")
		if self.z_corr == 1:
			self.apply(instr.INSTR_Z, q1)
			#print("Corr1 Z")
		yield self.run()
		
		
class BellMeasurementProgram(QuantumProgram):
	
	default_num_qubits = 2
	
	def program(self):
		q1, q2 = self.get_qubit_indices(2)
		#print(ns.qubits.reduced_dm(q1))
		self.apply(instr.INSTR_CNOT, [q1, q2])
		self.apply(instr.INSTR_H, q1)
		self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
		self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
		yield self.run()		

		
class BellMeasurementProtocol(NodeProtocol):
		
	def run(self):
		qubit_initialised = False
		entanglement_ready = False
		swapping_ended = False
		#qubit_init_program = InitStateProgram()
		measure_program = BellMeasurementProgram()
		
		while True:
			expr = yield (self.await_port_input(self.node.ports["qin_charlie"]) | self.await_port_input(self.node.ports["cin_bob"]))
			if expr.first_term.value:
				entanglement_ready = True
				#print("2")
			else:
				message = self.node.ports["cin_bob"].rx_input()
				m = message.items[0]
				if m == 1:
					swapping_ended = True
					print("A recibe ok de B")
            #yield self.await_port_input(self.node.ports["qin_charlie"])
            #entanglement_ready = True
            #swapping_ended = True
			if swapping_ended and entanglement_ready:
				# Once both qubits arrived, do BSM program and send to Bob
				#yield (self.await_signal(self.subprotocols["CorrectProtocol"], Signals.SUCCESS))
				#self.node.qmemory.execute_program(qubit_init_program)
				#yield self.await_program(self.node.qmemory)
				n = 1
				self.node.ports["cout_0"].tx_output(n)
				yield self.await_port_input(self.node.ports["qubitIN"])
				qubit_initialised = True
				if qubit_initialised:
					yield self.node.qmemory.execute_program(measure_program)
					m1, = measure_program.output["M1"]
					m2, = measure_program.output["M2"]
					#print("mensaje de Alice a Bob")
					self.node.ports["cout_bob"].tx_output((m1, m2))
					self.send_signal(Signals.SUCCESS)
					qubit_initialised = False
					entanglement_ready = False
					swapping_ended = False
					
					
class CreaQubit(NodeProtocol):
	
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["cin_alice"])
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0] #|0>
			self.node.ports["qubitOUT"].tx_output(qubit)
			
			
class CorrectionProtocol(NodeProtocol):
	
	def run(self):
		port_alice = self.node.ports["cin_alice"]
		port_charlie = self.node.ports["qin_charlie"]
		entanglement_ready = False
		meas_results = None
		while True:
			expr = yield (self.await_port_input(port_alice) | self.await_port_input(port_charlie))
			if expr.first_term.value:  # If measurements from Alice arrived
				meas_results, = port_alice.rx_input().items
			else:
				entanglement_ready = True
			if meas_results is not None and entanglement_ready:
				# Do corrections (blocking)
				if meas_results[0] == 1:
					self.node.qmemory.execute_instruction(instr.INSTR_Z)
					yield self.await_program(self.node.qmemory)
				if meas_results[1] == 1:
					self.node.qmemory.execute_instruction(instr.INSTR_X)
					yield self.await_program(self.node.qmemory)
				self.send_signal(Signals.SUCCESS, 0)
				entanglement_ready = False
				meas_results = None


def example_sim_setup(node_A, node_B):
	
	def collect_fidelity_data(evexpr):
		protocol = evexpr.triggered_events[-1].source
		mem_pos = protocol.get_signal_result(Signals.SUCCESS)
		qubit, = protocol.node.qmemory.pop(mem_pos)
		fidelity = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=True)
		global num
		num += 1
		#print(ns.qubits.reduced_dm(qubit))
		print(f"Fidelity: {fidelity}")
		qapi.discard(qubit)
		if num >= 10:
			ns.sim_stop()
		return {"fidelity": fidelity}

	protocol_alice = BellMeasurementProtocol(node_A)
	protocol_bob = CorrectionProtocol(node_B)
	dc = DataCollector(collect_fidelity_data)
	dc.collect_on(pydynaa.EventExpression(source=protocol_bob, event_type=Signals.SUCCESS.value))
	return protocol_alice, protocol_bob, dc
	
	
def run_experiment(num_runs, depolar_rate, distance=4e-3, dephase_rate=0.0):
	
	ns.sim_reset()
	network = example_network_setup(distance, depolar_rate, dephase_rate)
	node_a = network.get_node("Alice")
	node_b = network.get_node("Bob")
	node_r = network.get_node("Rep")
	node_0 = network.get_node("Nodo0")
	protocol_alice, protocol_bob, dc = example_sim_setup(node_a, node_b)
	protocol_bob2 = CorrectProtocol(node_b, 3)
	protocol_r = SwapProtocol(node_r, "swap")
	protocol_0 = CreaQubit(node_0)
	protocol_alice.start()
	protocol_bob.start()
	protocol_bob2.start()
	protocol_r.start()
	protocol_0.start()
	ns.sim_run(end_time=10, magnitude=1e9)
	return dc.dataframe
	
	
def create_plot(num_rep):
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	#dist = [2, 10, 15, 20, 25, 30, 40, 50]
	dist = [2, 5]
	#depolar_rates = [i for i in range(0, 200, 10)]
	for depolar_rate in [0]:
		data = pd.DataFrame()
		for distance in dist:
			#fallo = 1
			data[distance] = run_experiment(num_runs=num_rep, distance=distance, depolar_rate=depolar_rate, dephase_rate=0.0)['fidelity']
			global num
			print(f"num = {num}")
			num = 0
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		plot_style = {'kind': 'scatter', 'grid': True, 'title': "Fidelity of the teleported quantum state"}
		data.plot(y='fidelity', yerr='sem', label=f"{depolar_rate}", ax=ax)
		print(f"Depolarización: {depolar_rate}")
		print(f"{data}")
		print("")
	plt.xlabel("Distancia (km)")
	plt.ylabel("Fidelidad")
	plt.grid()
	plt.show()

num = 0
if __name__ == '__main__':
    create_plot(10)

	
		
		
		
		
		