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

def create_network(depol):
	network = Network("Red cuántica de Madrid")
	node_A = Node(name="A", qmemory = create_qprocessor("MemoriaA"))
	node_B = Node(name="B", qmemory = create_qprocessor("MemoriaB"))
	node_C = Node(name="C", qmemory = create_qprocessor("MemoriaC"))
	node_D = Node(name="D", qmemory = create_qprocessor("MemoriaD"))
	node_E = Node(name="E", qmemory = create_qprocessor("MemoriaE"))
	node_F = Node(name="F", qmemory = create_qprocessor("MemoriaF"))
	
	delay_model = PingPongDelayModel()
	
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
	
	network.add_nodes([node_A, node_B, node_C, node_D, node_E, node_F])
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
                      
class Protocol1(NodeProtocol):
	def __init__(self, node, qubit=None):
		super().__init__(node)
		self.qubit = qubit
		
	def run(self):
		if self.qubit is not None:
			#self.node.qmemory.put(qubits=self.qubit, positions=0, replace=True)
			self.node.ports["qubitOUT"].tx_output(self.qubit)
		while True:
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
			self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
			if self.node.qmemory.busy:
				yield self.await_program(self.node.qmemory)
			self.send_signal(Signals.SUCCESS)
			
def setup_datacollector(network, protocolo, nodo):
		
	def calc_fidelity(evexpr):
		qubit, = nodo.qmemory.peek([0]) #cogemos el qubit de la memoria del receptor
		global num
		global rep
		fidelity = ns.qubits.fidelity(qubit, ns.qubits.ketstates.s0, squared=True) #calculamos la fidelidad
		num += 1 #aumentamos el contador de fidelidades calculadas
		if num >= rep: #si ha calculado 2000 qubits paramos la simulación
			ns.sim_stop()
			print(f"{num}")
			num = 0
		return {"fidelity": fidelity} #devolvemos el resultado de la fidelidad
		
	dc = DataCollector(calc_fidelity, include_entity_name=False)
	dc.collect_on(py.EventExpression(source=protocolo, event_type=Signals.SUCCESS.value)) #calculamos la fidelidad cuando recibamos la señal del receptor
	return dc

num = 0
def run_simulation(depol, nodos):
	ns.sim_reset()
	network = create_network(depol)
	a_protocol = Protocol1(network.get_node("A"), qubit=None)
	if nodos==2: 
		b_protocol = Protocol3(network.get_node("B"))
		nodo = network.get_node("B")
		protocolo = b_protocol
	else: 
		b_protocol = Protocol2(network.get_node("B"))
	if nodos==3:
		c_protocol = Protocol3(network.get_node("C"))
		nodo = network.get_node("C")
		protocolo = c_protocol
	else:
		c_protocol = Protocol2(network.get_node("C"))
	if nodos==4:
		d_protocol = Protocol3(network.get_node("D"))
		nodo = network.get_node("D")
		protocolo = d_protocol
	else:
		d_protocol = Protocol2(network.get_node("D"))
	if nodos==5:
		e_protocol = Protocol3(network.get_node("E"))
		nodo = network.get_node("E")
		protocolo = e_protocol
	else:
		e_protocol = Protocol2(network.get_node("E"))
	f_protocol = Protocol3(network.get_node("F"))
	if nodos==6:
		nodo = network.get_node("F")
		protocolo = f_protocol
	
	dc = setup_datacollector(network, protocolo, nodo)
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
	global rep
	rep = num_rep
	#depolar = [i for i in range(500, 1100, 100)]
	depolar = [0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
	num_nodos = [2, 3, 4, 5, 6]
	ley = ["Comunicación A-B", "Comunicación A-C", "Comunicación A-D", "Comunicación A-E", "Comunicación A-F"]
	for nodos in num_nodos:
		data = pd.DataFrame()
		for depol in depolar:
			data[depol] = run_simulation(depol=depol, nodos=nodos)['fidelity']
		data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
		data.plot(y='fidelity', yerr='sem', label=ley[nodos-2], ax=ax)
		print(f"{data}")
		print("")
	plt.xlabel("Despolarización")
	plt.ylabel("Fidelidad")
	plt.title("Simulación de la red de Madrid")
	plt.grid()
	plt.show()
	
if __name__ == "__main__":
	create_plot(2000)
	
	