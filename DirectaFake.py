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
from netsquid.components.cchannel import *

from netsquid.components.instructions import *

from netsquid.nodes import *

from netsquid.components import *

from matplotlib import pyplot as plt

class EntanglingConnection(Connection):
	def __init__(self, length, source_frequency, name="EntanglingConnection"):
		super().__init__(name=name)
		qsource = QSource(f"qsource_{name}", StateSampler([ns.qubits.ketstates.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e9 / source_frequency), status=SourceStatus.INTERNAL)
		self.add_subcomponent(qsource, name="qsource")
		qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2, models={"delay_model": FibreDelayModel()})
		qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2, models={"delay_model": FibreDelayModel()})
		self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
		self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
		#Conectar el output al input del canal
		qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
		qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])
		
class ClassicalConnection(Connection):
	def __init__(self, length, name="ClassicalConnection"):
		super().__init__(name=name)
		self.add_subcomponent(ClassicalChannel("Channel_A2B", length=length, models={"delay_model": FibreDelayModel()}), forward_input=[("A", "send")], forward_output=[("B", "recv")])
		
def create_processor(depolar_rate, dephase_rate):
	measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate, time_independent=True)
	physical_instructions = [
		PhysicalInstruction(INSTR_INIT, duration=3, parallel=True),
      PhysicalInstruction(INSTR_H, duration=1, parallel=True, topology=[0, 1]),
      PhysicalInstruction(INSTR_X, duration=1, parallel=True, topology=[0]),
      PhysicalInstruction(INSTR_Z, duration=1, parallel=True, topology=[0]),
      PhysicalInstruction(INSTR_S, duration=1, parallel=True, topology=[0]),
      PhysicalInstruction(INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1)]),
      PhysicalInstruction(INSTR_MEASURE, duration=7, parallel=False, topology=[0], quantum_noise_model=measure_noise_model, apply_q_noise_after=False),
      PhysicalInstruction(INSTR_MEASURE, duration=7, parallel=False, topology=[1])]
	memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
	processor = QuantumProcessor("quantum_processor", num_positions=2, memory_noise_models=[memory_noise_model] * 2, phys_instructions=physical_instructions)
	return processor
      
def network_setup(node_distance=4e-3, depolar_rate=1e7, dephase_rate=0.2):
	#Conectamos a Alice y Bob con el procesador
	alice = Node("Alice", qmemory=create_processor(depolar_rate, dephase_rate))
	bob = Node("Bob", qmemory=create_processor(depolar_rate, dephase_rate))
	network = Network("Teleportation_network")
	network.add_nodes([alice, bob])
	c_conn = ClassicalConnection(length=node_distance)
	network.add_connection(alice, bob, connection=c_conn, label="Classical", port_name_node1="cout_bob", port_name_node2="cout_alice")
	#Conexión de pares entrelazados
	source_frequency = 4e4 / node_distance
	q_conn = EntanglingConnection(length=node_distance, source_frequency=source_frequency)
	port_ac, port_bc = network.add_connection(alice, bob, connection=q_conn, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charlie")
	alice.ports[port_ac].forward_input(alice.qmemory.ports['qin1'])
	bob.ports[port_bc].forward_input(bob.qmemory.ports['qin0'])
	return network
		
		
class InitStateProgram(QuantumProgram):
	default_num_qubits = 1
	def program(self):
		q1, = self.get_qubit_indices(1)
		self.apply(INSTR_INIT, q1)
		self.apply(INSTR_H, q1)
		self.apply(INSTR_S, q1)
		yield self.run()
		
class BellMeasurmentProgram(QuantumProgram):
	default_num_qubits = 2
	def program(self):
		q1, q2 = self.get_qubit_indices(2)
		self.apply(INSTR_CNOT, [q1, q2])
		self.apply(INSTR_H, q1)
		self.apply(INSTR_MEASURE, q1, output_key="M1")
		self.apply(INSTR_MEASURE, q2, output_key="M2")
		yield self.run()
      
class BellMeasurmentProtocol(NodeProtocol):
	def run(self):
		qubit_initialised = False
		entanglement_ready = False
		qubit_init_program = InitStateProgram()
		measure_program = BellMeasurmentProgram()
		self.node.qmemory.execute_program(qubit_init_program)
		while True:
			expr = yield(self.await_program(self.node.qmemory) | self.await_port_input(self.node.ports["qin_charlie"]))
			if expr.first_term.value:
				qubit_initialised = True
			else:
				entanglement_ready = True
			if qubit_initialised and entanglement_ready:
				yield self.node.qmemory.execute_program(measure_program)
				m1, = measure_program.output["M1"]
				m2, = measure_program.output["M2"]
				self.node.ports["cout_bob"].tx_output((m1, m2))
				self.send_signal(Signals.SUCCESS)
				qubit_initialised = False
				entanglement_ready = False
				self.node.qmemory.execute_program(qubit_init_program)
				
class CorrectProtocol(NodeProtocol):
	def run(self):
		port_alice = self.node.ports["cin_alice"]
		port_charlie = self.node.ports["qin_charlie"]
		entanglement_ready = False
		meas_result = None
		while True:
			expr = yield (self.await_port_input(port_alice) | self.await_port_input(port_charlie))
			if expr.first_term.value: #Si llega la medida de Alicia
				meas_result, = port_alice.rx_input().items
			else:
				entanglement_ready = True
			if meas_result is not None and entanglement_ready:
				if meas_result[0] == 1:
					self.node.qmemory.execute_instruction(INSTR_Z)
					yield self.await_program(self.node.qmemory)
				if meas_result[1] == 1:
					self.node.qmemory.execute_instruction(INSTR_X)
					yield self.await_program(self.node.qmemory)
				self.send_signal(Signals.SUCCESS, 0)
				entanglement_ready = False
				meas_result = None
				
def sim_setup(node_A, node_B):
	def collect_fidelity_data(evexpr):
		protocol = evexpr.triggered_events[-1].source
		mem_pos = protocol.get_signal_result(Signal.SUCCESS)
		qubit, = protocol.node.qmemory.pop(mem_pos)
		fidelity = qapi.fidelity(qubit, ns.y0, squared=True)
		qapi.discard(qubit)
		return {"fidelity": fidelity}
		
	protocol_alice = BellMeasurmentProtocol(node_A)
	protocol_bob = BellMeasurmentProtocol(node_B)
	dc = DataCollector(collect_fidelity_data)
	dc.collect_on(py.EventExpression(source=protocol_bob, event_type=Signals.SUCCESS.value))
	return protocol_alice, protocol_bob, dc
	
def run_experiment(num_runs, depolar_rate, distance=4e-3, dephase_rate=0.0):
	fidelity_data = pd.DataFrame()
	print("HOLA")
	ns.sim_reset()
	network = network_setup(distance, depolar_rate, dephase_rate)
	node_a = network.get_node("Alice")
	node_b = network.get_node("Bob")
	protocol_alice, protocol_bob, dc = sim_setup(node_a, node_b)
	protocol_alice.start()
	protocol_bob.start()
	q_conn = network.get_connection(node_a, node_b, label="quantum")
	cycle_runtime = (q_conn.subcomponents["qsource"].subcomponents["internal_clock"].models["timing_model"].delay)
	ns.sim_run(cycle_runtime * num_runs + 1)
	df = dc.dataframe
	df['depolar_rate'] = depolar_rate
	fidelity_data = fidelity_data.append(df)
	print("HOLA2")
	return fidelity_data
	
"""def create_plot():
	depolar_rates = [1e6 * i for i in range(0, 20, 10)]
	data = run_experiment(num_runs=1000, distance=4e-3, depolar_rates=depolar_rates, dephase_rate=0.0)
	data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
	plot_style = {'kind': 'scatter', 'grid': True, 'title': "Fidelity of the teleported quantum state"}
	data = data.groupby("depolar_rate")['fidelity'].agg(fidelity='mean', sem='sem').reset_index()
	print("HOLA3")
	data.plot(y='fidelity', yerr='sem', label=f"{depolar_rates}")
	#data.plot(x=0.5, y='fidelity', yerr='sem', **plot_style)
	plt.show()"""
	
def create_plot():
	fig, ax = plt.subplots()
	data = pd.DataFrame()
	for depolar_rate in [0, 20e6, 10e6]:
		try:
			data[depolar_rate] = run_experiment(num_runs=10, distance=4e-3, depolar_rate=depolar_rate, dephase_rate=0.0)['fidelity']
		except:
			print("HOLA3")
			data[depolar_rate] = 10
		plot_style = {'kind': 'scatter', 'grid': True, 'title': "Fidelity of the teleported quantum state"}
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		#data = data.groupby("depolar_rate")['fidelity'].agg(fidelity='mean', sem='sem').reset_index()
		data.plot(y='fidelity', yerr='sem', **plot_style, ax=ax)
	plt.show()
	
create_plot()
			
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		