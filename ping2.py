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
	network = Network("Ping network")
	node_ping = Node(name="Ping", qmemory = create_qprocessor("MemoriaPing"))
	node_pong = Node(name="Pong", qmemory = create_qprocessor("MemoriaPong"))
	delay_model = PingPongDelayModel()
	channel_1 = QuantumChannel(name="qchannel[ping to pong]", length=distance, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
	channel_2 = QuantumChannel(name="qchannel[pong to ping]", length=distance, models={"fibre_delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=None)})
	conn = DirectConnection(name="conn[ping|pong]",	channel_AtoB=channel_1, channel_BtoA=channel_2)
	channel_1.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
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
	noise_rate = 0
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
			qubits = ns.qubits.create_qubits(1)
			qubit = qubits[0] #|0>
			#meas, prob = ns.qubits.measure(qubit, ns.Z)
			#print("{} {}".format(meas, prob))
			self.node.ports["qubitIO"].tx_output(qubit)
			tiempo = ns.sim_time(ns.SECOND) + 1 #Esperamos 1 segundo para mandar el siguiente qubit
			yield self.await_timer(tiempo)
			
class PongProtocol(NodeProtocol):
	def __init__(self, node):
		super().__init__(node)
		
	def run(self):
		while True:
			yield self.await_port_input(self.node.ports["qubitIO"])
			message = self.node.ports["qubitIO"].rx_input()
			qubit = message.items[0]
			#meas, prob = ns.qubits.gmeasure(qubit, ops.Z.projectors)
			#meas, prob = ns.qubits.measure(qubit, ns.Z)
			#print("{} {}".format(meas, prob))
			self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
			#if meas == 1:
				#print("Corrección: {} {}".format(meas, prob))
				#self.node.qmemory.execute_instruction(instr.INSTR_X)
			if self.node.qmemory.busy:
				yield self.await_program(self.node.qmemory)
			self.send_signal(Signals.SUCCESS)
			
def setup_datacollector(network, protocol):
	nodoB = network.get_node("Pong")
	
	def calc_fidelity(evexpr):
		qubitB, = nodoB.qmemory.peek([0]) #cogemos el qubit de la memoria del receptor
		global num
		fidelity = ns.qubits.fidelity(qubitB, ns.qubits.ketstates.s0, squared=True) #calculamos la fidelidad
		num += 1 #aumentamos el contador de fidelidades calculadas
		if num >= 10: #si ha calculado 2000 qubits paramos la simulación
			ns.sim_stop()
			print(f"{num}")
			num = 0
		return {"fidelity": fidelity} #devolvemos el resultado de la fidelidad
		
	dc = DataCollector(calc_fidelity, include_entity_name=False)
	dc.collect_on(py.EventExpression(source=protocol, event_type=Signals.SUCCESS.value)) #calculamos la fidelidad cuando recibamos la señal del receptor
	return dc

num = 0	
def run_simulation(distance, att, depol):
	ns.sim_reset()
	network = create_network(distance, att, depol)
	ping_protocol = PingProtocol(network.get_node("Ping"), qubit=None)
	pong_protocol = PongProtocol(network.get_node("Pong"))
	dc = setup_datacollector(network, pong_protocol)
	ping_protocol.start()
	pong_protocol.start()
	ns.sim_run(end_time=10e9, magnitude=1e9)
	return dc.dataframe
	
def create_plot():
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	#att = 0
	#depolar = [i for i in range(500, 1100, 100)]
	depolar = [0.005, 0.0075, 0.01, 0.025, 0.05, 0.07, 0.1]
	aten = [0.3]
	#aten = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
	#depol = 0
	for att in aten:
		for depol in depolar:
			data = pd.DataFrame()
			for distance in [2, 5, 10, 15, 20, 25, 30, 40, 50]:
				data[distance] = run_simulation(distance=distance, att=att, depol=depol)['fidelity']
			data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
			data.plot(y='fidelity', yerr='sem', label=f"{att} dB/km - "f"{depol} ", ax=ax)
			print(f"Depolarización: {depol}")
			print(f"{data}")
			print("")
	plt.xlabel("Distancia (km)")
	plt.ylabel("Fidelidad")
	plt.title("Conexión directa")
	plt.legend(title='Atenuación y despolarización:')
	plt.grid()
	plt.show()
	
if __name__ == "__main__":
	create_plot()
	
	