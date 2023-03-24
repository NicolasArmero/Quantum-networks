import netsquid as ns
import pandas as pd
import numpy as np
import pydynaa as py


from netsquid.qubits import operators as ops

from netsquid.components import QuantumChannel
from netsquid.components.cchannel import ClassicalChannel
from netsquid.nodes import DirectConnection
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.qubits.state_sampler import StateSampler
from netsquid.components.models.qerrormodels import FibreLossModel, DepolarNoiseModel, DephaseNoiseModel
from netsquid.components.qprocessor import QuantumProcessor, PhysicalInstruction
from netsquid.protocols import NodeProtocol
from netsquid.nodes import Node, Connection, Network
from netsquid.components.qprogram import QuantumProgram
from netsquid.components.models.delaymodels import FixedDelayModel, FibreDelayModel
from netsquid.util.datacollector import DataCollector
from netsquid.components.models import QuantumErrorModel, DelayModel
import netsquid.components.instructions as instr
import netsquid.qubits.ketstates as ks
from netsquid.protocols.protocol import *

def create_network(depol, telep):
	network = Network("Red cuántica de Madrid")
	node_0 = Node(name="0", qmemory = create_processor("Memoria0"))
	node_A = Node(name="A", qmemory = create_processor("MemoriaA"))
	node_B = Node(name="B", qmemory = create_processor("MemoriaB"))
	node_C = Node(name="C", qmemory = create_processor("MemoriaC"))
	node_D = Node(name="D", qmemory = create_processor("MemoriaD"))
	node_E = Node(name="E", qmemory = create_processor("MemoriaE"))
	node_F = Node(name="F", qmemory = create_processor("MemoriaF"))
	
	delay_model = PingPongDelayModel()
	
	if telep == 1:
		channel_0A = QuantumChannel(name="qchannel[0 to A]", length=0.0001, models={"fibre_delay_model": FibreDelayModel()})
		channel_A0 = QuantumChannel(name="qchannel[A to 0]", length=0.0001, models={"fibre_delay_model": FibreDelayModel()})
		con0A = DirectConnection(name="conn[0|A]", channel_AtoB=channel_0A, channel_BtoA=channel_A0)
		
	if telep == 0:
		channel_AB = QuantumChannel(name="qchannel[A to B]", length=22.47, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(6.1/22.47), rng=np.random.RandomState(42))})
		channel_BA = QuantumChannel(name="qchannel[B to A]", length=22.47, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(6.1/22.47), rng=np.random.RandomState(42))})
		channel_BC = QuantumChannel(name="qchannel[B to C]", length=1.9, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(0.4/1.9), rng=np.random.RandomState(42))})
		channel_CB = QuantumChannel(name="qchannel[C to B]", length=1.9, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(0.4/1.9), rng=np.random.RandomState(42))})
		channel_CD = QuantumChannel(name="qchannel[C to D]", length=33.1, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(10.3/33.1), rng=np.random.RandomState(42))})
		channel_DC = QuantumChannel(name="qchannel[D to C]", length=33.1, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(10.3/33.1), rng=np.random.RandomState(42))})
		channel_DE = QuantumChannel(name="qchannel[D to E]", length=7.4, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(5.4/7.4), rng=np.random.RandomState(42))})
		channel_ED = QuantumChannel(name="qchannel[E to D]", length=7.4, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(5.4/7.4), rng=np.random.RandomState(42))})
		channel_EF = QuantumChannel(name="qchannel[E to F]", length=24.2, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(6/24.2), rng=np.random.RandomState(42))})
		channel_FE = QuantumChannel(name="qchannel[F to E]", length=24.2, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=(6/24.2), rng=np.random.RandomState(42))})
	
		conAB = DirectConnection(name="conn[A|B]", channel_AtoB=channel_AB, channel_BtoA=channel_BA)
		conBC = DirectConnection(name="conn[B|C]", channel_AtoB=channel_BC, channel_BtoA=channel_CB)
		conCD = DirectConnection(name="conn[C|D]", channel_AtoB=channel_CD, channel_BtoA=channel_DC)
		conDE = DirectConnection(name="conn[D|E]", channel_AtoB=channel_DE, channel_BtoA=channel_ED)
		conEF = DirectConnection(name="conn[E|F]", channel_AtoB=channel_EF, channel_BtoA=channel_FE)
	
		channel_AB.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_BA.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_BC.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_CB.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_CD.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_DC.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_DE.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_ED.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_EF.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		channel_FE.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
	
		node_A.connect_to(remote_node=node_B, connection=conAB, local_port_name="qubitOUT", remote_port_name="qubitIN")
		node_B.connect_to(remote_node=node_C, connection=conBC, local_port_name="qubitOUT", remote_port_name="qubitIN")
		node_C.connect_to(remote_node=node_D, connection=conCD, local_port_name="qubitOUT", remote_port_name="qubitIN")
		node_D.connect_to(remote_node=node_E, connection=conDE, local_port_name="qubitOUT", remote_port_name="qubitIN")
		node_E.connect_to(remote_node=node_F, connection=conEF, local_port_name="qubitOUT", remote_port_name="qubitIN")
		
	network.add_nodes([node_0, node_A, node_B, node_C, node_D, node_E, node_F])
	
	if telep == 1:
		port_0a, port_a0 = network.add_connection(node_0, node_A, connection=conA0, label="quantum", port_name_node1="qubitOUT", port_name_node2="qubitIN")
		node_A.ports[port_a0].forward_input(node_C.qmemory.ports['qin0'])
		c_A0 = ClassicalConnection(length=0.0001)
		network.add_connection(node_A, node_0, connection=c_A0, label="classical", port_name_node1="cout_0", port_name_node2="cin_A")
		c_AB = ClassicalConnection(length=22.47)
		network.add_connection(node_A, node_B, connection=c_AB, label="classical", port_name_node1="cout", port_name_node2="cin")
		source_frequency = 4e4 / 33.1
		q_AB = EntanglingConnection(length=22.47, source_frequency=source_frequency)
		for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
			q_AB.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		port_ach, port_bchl = network.add_connection(node_A, node_B, connection=q_AB, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charliel")
		node_A.ports[port_ach].forward_input(node_A.qmemory.ports['qin1'])
		node_B.ports[port_bchl].forward_input(node_B.qmemory.ports['qin0'])
		
		c_BC = ClassicalConnection(length=1.9)
		network.add_connection(node_B, node_C, connection=c_BC, label="classical", port_name_node1="cout", port_name_node2="cin")
		source_frequency = 4e4 / 33.1
		q_BC = EntanglingConnection(length=1.9, source_frequency=source_frequency)
		for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
			q_BC.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		port_bch, port_cchl = network.add_connection(node_B, node_C, connection=q_BC, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charliel")
		node_B.ports[port_bch].forward_input(node_B.qmemory.ports['qin1'])
		node_C.ports[port_cchl].forward_input(node_C.qmemory.ports['qin0'])
		
		#c_DA = ClassicalConnection(length=41)
		#network.add_connection(node_D, node_A, connection=c_DA, label="classical", port_name_node1="cout_A", port_name_node2="cin_D")
		c_CD = ClassicalConnection(length=33.1)
		network.add_connection(node_C, node_D, connection=c_CD, label="classical", port_name_node1="cout_D", port_name_node2="cin_C")
		# Setup entangling connection between nodes:
		source_frequency = 4e4 / 33.1
		q_CD = EntanglingConnection(length=33.1, source_frequency=source_frequency)
		for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
			q_CD.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		port_cch, port_dchl = network.add_connection(node_C, node_D, connection=q_conn, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charliel")
		node_C.ports[port_cch].forward_input(node_C.qmemory.ports['qin1'])
		node_D.ports[port_dchl].forward_input(node_D.qmemory.ports['qin0'])
		
		c_DE = ClassicalConnection(length=7.4)
		network.add_connection(node_D, node_E, connection=c_AB, label="classical", port_name_node1="cout", port_name_node2="cin")
		source_frequency = 4e4 / 33.1
		q_DE = EntanglingConnection(length=7.4, source_frequency=source_frequency)
		for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
			q_DE.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		port_dch, port_echl = network.add_connection(node_D, node_E, connection=q_AB, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charliel")
		node_D.ports[port_dch].forward_input(node_D.qmemory.ports['qin1'])
		node_E.ports[port_echl].forward_input(node_E.qmemory.ports['qin0'])
		
		c_EF = ClassicalConnection(length=24.2)
		network.add_connection(node_E, node_F, connection=c_EF, label="classical", port_name_node1="cout", port_name_node2="cin")
		source_frequency = 4e4 / 33.1
		q_EF = EntanglingConnection(length=24.2, source_frequency=source_frequency)
		for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
			q_EF.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
		port_ech, port_fchl = network.add_connection(node_E, node_F, connection=q_EF, label="quantum", port_name_node1="qin_charlie", port_name_node2="qin_charliel")
		node_E.ports[port_ech].forward_input(node_E.qmemory.ports['qin1'])
		node_F.ports[port_fchl].forward_input(node_F.qmemory.ports['qin0'])
	
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
	
def create_processor(name):
	depolar_rate = 0
	dephase_rate = 0
	measure_noise_model = DephaseNoiseModel(dephase_rate=dephase_rate, time_independent=True)
	physical_instructions = [
        PhysicalInstruction(instr.INSTR_INIT, duration=3, parallel=True),
        PhysicalInstruction(instr.INSTR_H, duration=1, parallel=True, topology=[0, 1]),
        PhysicalInstruction(instr.INSTR_X, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_Z, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_S, duration=1, parallel=True, topology=[0]),
        PhysicalInstruction(instr.INSTR_CNOT, duration=4, parallel=True, topology=[(0, 1)]),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[0], q_noise_model=measure_noise_model, apply_q_noise_after=False),
        PhysicalInstruction(instr.INSTR_MEASURE, duration=7, parallel=False, topology=[1])
    ]
	memory_noise_model = DepolarNoiseModel(depolar_rate=depolar_rate)
	processor = QuantumProcessor("quantum_processor", num_positions=2, memory_noise_models=[memory_noise_model] * 2, phys_instructions=physical_instructions)
	return processor
    
class EntanglingConnection(Connection):
	
	def __init__(self, length, source_frequency, name="EntanglingConnection"):
		super().__init__(name=name)
		qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e10 / source_frequency), status=SourceStatus.INTERNAL)
		self.add_subcomponent(qsource, name="qsource")
		qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0.1, rng=np.random.RandomState(42))})
		qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=0.1, rng=np.random.RandomState(42))})
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
            #ns.qubits.apply_dda_noise(qubit, depol=prob, deph=0.5, ampl=0.1)
            #ns.qubits.qubitapi.apply_pauli_noise(qubit, (0, prob, 0, (1-prob)))
            #ns.qubits.dephase(qubit, prob=prob)
                      
class Protocol1(NodeProtocol):
	
	def __init__(self, node, telep):
		super().__init__(node)
		self.telep = telep
		
	def run(self):
		while True:
			if self.telep:
				yield self.await_port_input(self.node.ports["cin_A"])
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0] #|0>
			self.node.ports["qubitOUT"].tx_output(qubit)
			tiempo = ns.sim_time(ns.SECOND) + 1 #Esperamos 1 segundo para mandar el siguiente qubit
			yield self.await_timer(tiempo)

class Protocol2(NodeProtocol):
	
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitIN"])
			message = self.node.ports["qubitIN"].rx_input()
			qubit = message.items[0]
			self.node.ports["qubitOUT"].tx_output(qubit)
						
class Protocol3(NodeProtocol):
	
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitIN"])
			message = self.node.ports["qubitIN"].rx_input()
			qubit = message.items[0]
			if qubit is not None:
				#print("llega")
				self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
				#print("se va")
				if self.node.qmemory.busy:
					yield self.await_program(self.node.qmemory)
				self.send_signal(Signals.SUCCESS)
			
class BellMeasurementProgram(QuantumProgram):
	
	default_num_qubits = 2

	def program(self):
		q1, q2 = self.get_qubit_indices(2)
		self.apply(instr.INSTR_CNOT, [q1, q2])
		self.apply(instr.INSTR_H, q1)
		self.apply(instr.INSTR_MEASURE, q1, output_key="M1")
		self.apply(instr.INSTR_MEASURE, q2, output_key="M2")
		yield self.run()
		
class BellMeasurementProtocol(NodeProtocol):
	
	def run(self):
		port_B = self.node.ports["qubitIN"]
		port_D = self.node.ports["cout"]
		qubit_B = False
		entanglement_ready = False
		measure_program = BellMeasurementProgram()
		while True:
			expr = yield (self.await_port_input(port_B) | self.await_port_input(self.node.ports["qin_charlie"]))
			if expr.first_term.value:
				qubit_B = True
				#print("Qubit de alice")
			else:
				entanglement_ready = True
				#print("qubit charlie medio")
			if qubit_B and entanglement_ready:
				# Once both qubits arrived, do BSM program and send to Bob
				yield self.node.qmemory.execute_program(measure_program)
				m1, = measure_program.output["M1"]
				m2, = measure_program.output["M2"]
				port_D.tx_output((m1, m2))
				#print("Envia medio a B")
				self.send_signal(Signals.SUCCESS)
				qubit_B = False
				entanglement_ready = False
				
class CorrectionProtocol(NodeProtocol):
	
	def run(self):
		port_C = self.node.ports["cin"]
		port_charlie = self.node.ports["qin_charliel"]
		entanglement_ready = False
		meas_results = None
		while True:
			# Wait for measurement results of Alice or qubit from Charlie to arrive
			n = 0
			self.node.ports["cout_A"].tx_output(n)
			expr = yield (self.await_port_input(port_C) | self.await_port_input(port_charlie))
			if expr.first_term.value:
				# If measurements from Alice arrived
				meas_results, = port_C.rx_input().items
				#print("Recibe B")
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
				#print("LLEGA")
				qubit, = self.node.qmemory.pop(0)
				#print("Sale")
				self.node.ports["qubitOUT"].tx_output(qubit)
				entanglement_ready = False
				meas_results = None
			
def setup_datacollector(network, protocol, rep):
	nodo = network.get_node("F")
	
	def calc_fidelity(evexpr):
		qubit, = nodo.qmemory.peek([0]) #cogemos el qubit de la memoria del receptor
		global num
		fidelity = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=True) #calculamos la fidelidad
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
def run_simulation(depol, num_rep, telep):
	ns.sim_reset()
	network = create_network(depol, telep)
	if telep == 0:
		a_protocol = Protocol1(network.get_node("A"), telep)
		b_protocol = Protocol2(network.get_node("B"))
		c_protocol = Protocol2(network.get_node("C"))
		d_protocol = Protocol2(network.get_node("D"))
		e_protocol = Protocol2(network.get_node("E"))
		f_protocol = Protocol3(network.get_node("F"))
	elif telep == 1:
		0_protocol = Protocol1(network.get_node("0"), telep)
		a_protocol = BellMeasurementProtocol(network.get_node("A"))
		f_protocol = CorrectionProtocol(network.get_node("F"))
	
	dc = setup_datacollector(network, f_protocol, num_rep)
	a_protocol.start()
	b_protocol.start()
	c_protocol.start()
	d_protocol.start()
	e_protocol.start()
	f_protocol.start()
	
	ns.sim_run(end_time=10e9, magnitude=1e9)
	return dc.dataframe
	
def create_plot(num_rep=100):
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	#depolar = [i for i in range(500, 1100, 100)]
	depolar = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
	#depolar = [0.001]
	data = pd.DataFrame()
	for telep in [0, 1]:
		data = pd.DataFrame()
		for depol in depolar:
			data[depol] = run_simulation(depol=depol, num_rep=num_rep, telep=telep)['fidelity']
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		if telep == 0:
			print("directa")
			data.plot(y='fidelity', yerr='sem', label="Comunicación C-D directa", ax=ax)
		else:
			print("telep")
			data.plot(y='fidelity', yerr='sem', label="Comunicación C-D teleportación", ax=ax)
		print(f"{data}")
		print("")
	plt.xlabel("Despolarización")
	plt.ylabel("Fidelidad")
	plt.title("Simulación de la red de Madrid")
	plt.grid()
	plt.show()
	
if __name__ == "__main__":
	create_plot(1000)
	
	