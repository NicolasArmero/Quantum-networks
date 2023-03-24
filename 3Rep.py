import netsquid as ns
import pandas as pd
import numpy as np
import pydynaa as py

import netsquid.qubits.qubitapi as qapi

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
from netsquid.components.cchannel import ClassicalChannel
from netsquid.nodes.connections import *
from netsquid.protocols.nodeprotocols import *
from netsquid.util.datacollector import *

from netsquid.components.instructions import *

from netsquid.nodes import *

from netsquid.components import *

class EntanglingConnection(Connection):
	
	def __init__(self, length, source_frequency, name="EntanglingConnection"):
		super().__init__(name=name)
		qsource = QSource(f"qsource_{name}", StateSampler([ns.qubits.ketstates.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e9/source_frequency), status=SourceStatus.INTERNAL)
		self.add_subcomponent(qsource, name="qsource")
		qchannel_c2a = QuantumChannel("qchannel_C2A", length=length/2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0.25, rng=np.random.RandomState(42))})
		qchannel_c2b = QuantumChannel("qchannel_C2B", length=length/2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0.25, rng=np.random.RandomState(42))})
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


class FibreDepolarizeModel(QuantumErrorModel):
	
	def __init__(self, p_depol_init=0, p_depol_length=0.025):
		super().__init__()
		self.properties['p_depol_init'] = p_depol_init
		self.properties['p_depol_length'] = p_depol_length
		self.required_properties = ['length']

	def error_operation(self, qubits, delta_time=0, **kwargs):
		for qubit in qubits:
			prob = 1 - (1 - self.properties['p_depol_init']) * np.power(10, - kwargs['length']**2 * self.properties['p_depol_length'] / 10)
			ns.qubits.depolarize(qubit, prob=prob)
			
			
def create_processor(depolar_rate, dephase_rate):
    
    # We'll give both Alice and Bob the same kind of processor
    measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate, time_independent=True)
    physical_instructions = [
        PhysicalInstruction(INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(INSTR_H, duration=1, parallel=True, topology=[0, 1]),
        PhysicalInstruction(INSTR_X, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(INSTR_Z, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(INSTR_S, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(INSTR_MEASURE, duration=7, parallel=False, topology=[0], q_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(INSTR_MEASURE, duration=7, parallel=False, topology=[1]),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=1)
    ]
    memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
    processor = QuantumProcessor("quantum_processor", num_positions=2, memory_noise_models=[memory_noise_model] * 2, phys_instructions=physical_instructions)
    return processor

def create_qprocessor(name):
    
    noise_rate = 0
    gate_duration = 1
    gate_noise_model = DephaseNoiseModel(noise_rate)
    mem_noise_model = DepolarNoiseModel(noise_rate)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=gate_duration, q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_Z, duration=gate_duration, q_noise_model=gate_noise_model),
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=gate_duration),
    ]
    qproc = QuantumProcessor(name, num_positions=2, fallback_to_nonphysical=False, mem_noise_models=[mem_noise_model] * 2, phys_instructions=physical_instructions)
    return qproc
    

def example_network_setup(node_distance, depol):
	
	network = Network("Repetidores con nodo antes")
	
	nodo0 = Node("0", qmemory=create_qprocessor("mem_0"))
	alice = Node("Alice", qmemory=create_processor(0, 0))
	bob = Node("Bob", qmemory=create_qprocessor("mem_b"))
	rep = Node("Rep", qmemory=create_qprocessor("mem_r"))
	
	channel_1 = QuantumChannel(name="qchannel02A", length=node_distance, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0, rng=np.random.RandomState(42))})
	channel_2 = QuantumChannel(name="qchannelA20", length=node_distance, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0, rng=np.random.RandomState(42))})
	conn = DirectConnection(name="conn0A",	channel_AtoB=channel_1, channel_BtoA=channel_2)
	
	network.add_nodes([nodo0, alice, bob, rep])
	
	port_0a, port_a0 = network.add_connection(nodo0, alice, connection=conn, label="quantum", port_name_node1="qubitOUT", port_name_node2="qubitIN")
	alice.ports[port_a0].forward_input(alice.qmemory.ports['qin0'])
	
	source_frequency = 4e4 / node_distance
	qcon_a = EntanglingConnection(name="qcon_A-R", length=node_distance/2, source_frequency=source_frequency)
	for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
		qcon_a.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	port_ac, port_rac = network.add_connection(alice, rep, connection=qcon_a, label="quantum", port_name_node1="qin_ca", port_name_node2="qin_cra")
	alice.ports[port_ac].forward_input(alice.qmemory.ports["qin1"])
	rep.ports[port_rac].forward_input(rep.qmemory.ports["qin0"])
	
	qcon_b = EntanglingConnection(name="qcon_B-R", length=node_distance/2, source_frequency=source_frequency)
	for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
		qcon_b.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	port_rbc, port_bc = network.add_connection(rep, bob, connection=qcon_b, label="quantum", port_name_node1="qin_crb", port_name_node2="qin_cb")
	rep.ports[port_rbc].forward_input(rep.qmemory.ports["qin1"])
	bob.ports[port_bc].forward_input(bob.qmemory.ports["qin0"])
	
	c_conn = ClassicalConnection(length=node_distance)
	network.add_connection(bob, nodo0, connection=c_conn, label="classical", port_name_node1="cout_0", port_name_node2="cin_bob")
	
	ccon_rb = ClassicalConnection(name="ccon_RB", length=node_distance/2)
	port_r_out_b, port_b_in_r = network.add_connection(rep, bob, connection=ccon_rb, label="classical", port_name_node1="cout_bob", port_name_node2="cin_rep")
	ccon_br = ClassicalConnection(name="ccon_BR", length=node_distance/2)
	port_r_out_b, port_b_in_r = network.add_connection(bob, rep, connection=ccon_br, label="classical", port_name_node1="cout_rep", port_name_node2="cin_bob")
	
	ccon_ab = ClassicalConnection(name="ccon_AB", length=node_distance)
	network.add_connection(alice, bob, connection=ccon_ab, label="classical", port_name_node1="port_a2b", port_name_node2="port_inba")
	ccon_ba = ClassicalConnection(name="ccon_BA", length=node_distance)
	network.add_connection(bob, alice, connection=ccon_ba, label="classical", port_name_node1="port_b2a", port_name_node2="port_inab")
	
	return network
	
	
def setup_repeater_protocol(network):
	
	protocol = LocalProtocol(nodes=network.nodes)
	
	nodo0 = network.get_node("0")
	alice = network.get_node("Alice")
	bob = network.get_node("Bob")
	rep = network.get_node("Rep")
	
	subprotocol = CeroProtocol(node=nodo0)
	protocol.add_subprotocol(subprotocol)
	subprotocol = SwapProtocol(node=rep, name="Swap protocol")
	protocol.add_subprotocol(subprotocol)
	# Add CorrectProtocol to Bob
	subprotocol = CorrectProtocol(bob, 3)
	protocol.add_subprotocol(subprotocol)
	# Add BellMeasurementProtocol to Alice
	subprotocol = BellMeasurementProtocol(alice)
	protocol.add_subprotocol(subprotocol)
	# Add CorrectionProtocol to Bob
	subprotocol = CorrectionProtocol(bob)
	protocol.add_subprotocol(subprotocol)
	
	return protocol
	
	
class CeroProtocol(NodeProtocol):
	
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["cin_bob"])
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0] #|0>
			if self.node.qmemory.busy:
				yield self.await_program(self.node.qmemory)
			self.node.ports["qubitOUT"].tx_output(qubit)
			tiempo = ns.sim_time(ns.SECOND) + 1 #Esperamos 1 segundo para mandar el siguiente qubit
			yield self.await_timer(tiempo)
			
			
class SwapProtocol(NodeProtocol):
	
	def __init__(self, node, name):
		super().__init__(node, name)
		self._qmem_input_port_l = self.node.qmemory.ports["qin0"]
		self._qmem_input_port_r = self.node.qmemory.ports["qin1"]
		self._program = QuantumProgram(num_qubits=2)
		q1, q2 = self._program.get_qubit_indices(num_qubits=2)
		self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)
		
	def run(self):
		while True:
			yield (self.await_port_input(self._qmem_input_port_l) & self.await_port_input(self._qmem_input_port_r))
			# Perform Bell measurement
			yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
			m, = self._program.output["m"]
			#Manda resultado del Swap a Bob
			self.node.ports["cout_bob"].tx_output(Message(m))
			
			
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
			yield self.await_port_input(self.node.ports["cin_rep"]) #Espera hasta que recibe un qubit
			message = self.node.ports["cin_rep"].rx_input()
			if message is None or len(message.items) != 1:
				continue
			m = message.items[0]
			if m == ns.qubits.ketstates.BellIndex.B01 or m == ns.qubits.ketstates.BellIndex.B11:
				self._x_corr += 1
			if m == ns.qubits.ketstates.BellIndex.B10 or m == ns.qubits.ketstates.BellIndex.B11:
				self._z_corr += 1
			self._counter += 1
			if self._counter == self.num_nodes - 2: #Si es el nodo final y no un repetidor
				if self._x_corr or self._z_corr:
					self._program.set_corrections(self._x_corr, self._z_corr)
					yield self.node.qmemory.execute_program(self._program, qubit_mapping=[0]) #Llama a la función de abajo
				#self.send_signal(Signals.SUCCESS)
				n = 1
				self.node.ports["port_b2a"].tx_output(n)
				#self.node.ports["cout_rep"].tx_output(n)
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
			self.apply(INSTR_X, q1)
		if self.z_corr == 1:
			self.apply(INSTR_Z, q1)
		yield self.run()
       		
		
class BellMeasurementProgram(QuantumProgram):
	
	default_num_qubits = 2
	
	def program(self):
		q1, q2 = self.get_qubit_indices(2)
		self.apply(INSTR_CNOT, [q1, q2])
		self.apply(INSTR_H, q1)
		self.apply(INSTR_MEASURE, q1, output_key="M1")
		self.apply(INSTR_MEASURE, q2, output_key="M2")
		yield self.run()
		
		
class BellMeasurementProtocol(NodeProtocol):
	
	def run(self):
		entanglement_ready = False
		swapping_ended = False
		port0 = self.node.ports["qubitIN"]
		measure_program = BellMeasurementProgram()
				
		while True:
			expr = yield (self.await_port_input(port0) | self.await_port_input(self.node.ports["port_inab"]))
			#expr = yield (self.await_port_input(self.node.ports["qin_ca"]) | self.await_port_input(self.node.ports["port_inab"]))
			if expr.first_term.value:
				entanglement_ready = True
			else:
				swapping_ended = True
			if swapping_ended and entanglement_ready:
				yield self.node.qmemory.execute_program(measure_program)
				m1, = measure_program.output["M1"]
				m2, = measure_program.output["M2"]
				#m1 = 0
				#m2 = 0
				self.node.ports["port_a2b"].tx_output((m1, m2))
				self.send_signal(Signals.SUCCESS)
				entanglement_ready = False
				swapping_ended = False
				
				
class CorrectionProtocol(NodeProtocol):
	
	def run(self):
		port_alice = self.node.ports["port_inba"]
		port_charlie = self.node.ports["qin_cb"]
		meas_results = None
		entanglement_ready = False
		
		while True:
			n = 0
			self.node.ports["cout_0"].tx_output(n)
			expr = yield (self.await_port_input(port_alice) | self.await_port_input(port_charlie))
			if expr.first_term.value:
				meas_results, = port_alice.rx_input().items
			else:
				entanglement_ready = True
			if meas_results is not None and entanglement_ready:
				if meas_results[0] == 1:
					self.node.qmemory.execute_instruction(INSTR_Z)
					yield self.await_program(self.node.qmemory)
				if meas_results[1] == 1:
					self.node.qmemory.execute_instruction(INSTR_X)
					yield self.await_program(self.node.qmemory)
				self.send_signal(Signals.SUCCESS, 0)
				meas_results = None
				entanglement_ready = False
				
def setup_datacollector(network, protocol, num_runs):
		
	def calc_fidelity(evexpr):
		prot = evexpr.triggered_events[-1].source
		mem_pos = protocol.get_signal_result(Signals.SUCCESS)
		qubit, = prot.node.qmemory.pop([0])
		fidelity = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=True)
		global num
		num += 1
		qapi.discard(qubit)
		if num >= num_runs:
			ns.sim_stop()
		return {"fidelity": fidelity}
		
	dc = DataCollector(calc_fidelity)
	dc.collect_on(py.EventExpression(source=protocol.subprotocols['CorrectionProtocol'], event_type=Signals.SUCCESS.value))
	return dc
	
	
def run_simulation(distance, depolar, num_runs):
	
	ns.sim_reset()
	est_runtime = (0.5 + 3 - 1) * distance * 5e3
	network = example_network_setup(distance, depolar)
	protocol = setup_repeater_protocol(network)
	dc = setup_datacollector(network, protocol, num_runs)
	protocol.start()
	ns.sim_run(end_time=10e9, magnitude=1e9)
	return dc.dataframe
	
def create_plot(num_runs):
	
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	dist = [2, 5, 10, 15, 20, 25, 30, 40, 50]
	for depolar in [0]:
		data = pd.DataFrame()
		for distance in dist:
			data[distance] = run_simulation(distance, depolar, num_runs)['fidelity']
			global num
			print(f"num = {num}")
			num = 0
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		data.plot(y='fidelity', yerr='sem', label=f"{depolar}", ax=ax)
		print(f"Depolarización: {depolar}")
		print(f"{data}")
		print("")
		
	plt.xlabel("distance")
	plt.ylabel("Fidelidad")
	plt.grid()
	plt.legend(title='Atenuación y número de repetidores:')
	plt.title("Teleportación con repetidores")
	plt.show()
	
num = 0	
if __name__ == "__main__":
    create_plot(10)




















