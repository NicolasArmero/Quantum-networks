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

def directa(distance, att, depol):
	ns.sim_reset()
	network = create_network(distance, att, depol)
	ping_protocol = PingProtocol(network.get_node("Ping"), qubit=None)
	pong_protocol = PongProtocol(network.get_node("Pong"))
	dc = setup_datacollector_d(network, pong_protocol)
	ping_protocol.start()
	pong_protocol.start()
	ns.sim_run(end_time=10e9, magnitude=1e9)
	return dc.dataframe
    
def create_network(distance, att, depol):
    network = Network("Ping network")
    node_ping = Node(name="Ping", qmemory = create_qprocessor("MemoriaPing"))
    node_pong = Node(name="Pong", qmemory = create_qprocessor("MemoriaPong"))
    #delay_model = PingPongDelayModel()
    channel_1 = QuantumChannel(name="qchannel[ping to pong]", length=distance, models={"fibre_depolarize_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
    channel_2 = QuantumChannel(name="qchannel[pong to ping]", length=distance, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=None)})
    conn = DirectConnection(name="conn[ping|pong]",	channel_AtoB=channel_1, channel_BtoA=channel_2)
    channel_1.models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
    #network.add_connection(node_ping, node_pong, connection=conn)
    node_ping.connect_to(remote_node=node_pong, connection=conn, local_port_name="qubitIO", remote_port_name="qubitIO")
    network.add_nodes([node_ping, node_pong])
    return network
        
def create_qprocessor(name):
    noise_rate = 200
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
			self.node.qmemory.put(qubits=qubit, positions=0, replace=True)
			#if qubit != ([[1.+0.j], [0.+0.j]]):
				#self.node.qmemory.execute_instruction(instr.INSTR_Z)
			if self.node.qmemory.busy:
				yield self.await_program(self.node.qmemory)
			self.send_signal(Signals.SUCCESS)            
        
def setup_datacollector_d(network, protocol):
	nodoB = network.get_node("Pong")
	
	def calc_fidelity(evexpr):
		qubitB, = nodoB.qmemory.peek([0]) #cogemos el qubit de la memoria del receptor
		global num
		global rep
		fidelity = ns.qubits.fidelity(qubitB, ns.qubits.ketstates.s0, squared=True) #calculamos la fidelidad
		num += 1 #aumentamos el contador de fidelidades calculadas
		if num >= rep: #si ha calculado 2000 qubits paramos la simulación
			ns.sim_stop()
			print(f"{num}")
			num = 0
		return {"fidelity": fidelity} #devolvemos el resultado de la fidelidad
		
	dc = DataCollector(calc_fidelity, include_entity_name=False)
	dc.collect_on(py.EventExpression(source=protocol, event_type=Signals.SUCCESS.value)) #calculamos la fidelidad cuando recibamos la señal del receptor
	return dc      
       
#CÓDIGO DE LA TRANSMISIÓN CON REPETIDORES

def setup_network(num_nodes, node_distance, source_frequency, att, depol):
    if num_nodes < 3:
        raise ValueError(f"Can't create repeater chain with {num_nodes} nodes.")
    network = Network("Repeater_chain_network")
    # Create nodes with quantum processors
    nodes = []
    for i in range(num_nodes):
        # Prepend leading zeros to the number
        num_zeros = int(np.log10(num_nodes)) + 1
        nodes.append(Node(f"Node_{i:0{num_zeros}d}", qmemory=create_qprocessor(f"qproc_{i}")))
    network.add_nodes(nodes)
    # Create quantum and classical connections:
    for i in range(num_nodes - 1):
        node, node_right = nodes[i], nodes[i+1]
        # Create quantum connection
        qconn = EntanglingConnection(name=f"qconn_{i}-{i+1}", length=node_distance, source_frequency=source_frequency, att=att)
        # Add a noise model which depolarizes the qubits exponentially
        # depending on the connection length
        for channel_name in ['qchannel_C2A', 'qchannel_C2B']:
            qconn.subcomponents[channel_name].models['quantum_noise_model'] = FibreDepolarizeModel(p_depol_init=0, p_depol_length=depol)
        port_name, port_r_name = network.add_connection(node, node_right, connection=qconn, label="quantum")
        # Forward qconn directly to quantum memories for right and left inputs:
        node.ports[port_name].forward_input(node.qmemory.ports["qin0"])  # R input
        node_right.ports[port_r_name].forward_input(node_right.qmemory.ports["qin1"])  # L input
        # Create classical connection
        cconn = ns.examples.teleportation.ClassicalConnection(name=f"cconn_{i}-{i+1}", length=node_distance)
        port_name, port_r_name = network.add_connection(node, node_right, connection=cconn, label="classical", port_name_node1="ccon_R", port_name_node2="ccon_L")
        # Forward cconn to right most node
        if "ccon_L" in node.ports:
            node.ports["ccon_L"].bind_input_handler(lambda message, _node=node: _node.ports["ccon_R"].tx_output(message))
    return network  

class EntanglingConnection(Connection):
    
    def __init__(self, length, source_frequency, att, name="EntanglingConnection"):
        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}", StateSampler([ns.qubits.ketstates.b00], [1.0]), num_ports=2, timing_model=FixedDelayModel(delay=1e9 / source_frequency), status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource, name="qsource")
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length/2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length/2, models={"delay_model": FibreDelayModel(), "quantum_loss_model": FibreLossModel(p_loss_init=0, p_loss_length=att, rng=np.random.RandomState(42))})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"]) 
        
def setup_repeater_protocol(network):
    protocol = LocalProtocol(nodes=network.nodes)
    # Add SwapProtocol to all repeater nodes. Note: we use unique names,
    # since the subprotocols would otherwise overwrite each other in the main protocol.
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]
    for node in nodes[1:-1]:
        subprotocol = SwapProtocol(node=node, name=f"Swap_{node.name}")
        protocol.add_subprotocol(subprotocol)
    # Add CorrectProtocol to Bob
    subprotocol = CorrectProtocol(nodes[-1], len(nodes))
    protocol.add_subprotocol(subprotocol)
    return protocol

class SwapProtocol(NodeProtocol):
    def __init__(self, node, name):
        super().__init__(node, name)
        self._qmem_input_port_l = self.node.qmemory.ports["qin1"]
        self._qmem_input_port_r = self.node.qmemory.ports["qin0"]
        self._program = QuantumProgram(num_qubits=2)
        q1, q2 = self._program.get_qubit_indices(num_qubits=2)
        self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", inplace=False)

    def run(self):
        while True:
            yield (self.await_port_input(self._qmem_input_port_l) & self.await_port_input(self._qmem_input_port_r))
            # Perform Bell measurement
            yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1, 0])
            m, = self._program.output["m"]
            # Send result to right node on end
            self.node.ports["ccon_R"].tx_output(Message(m))
            
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
            yield self.await_port_input(self.node.ports["ccon_L"]) #Espera hasta que recibe un qubit
            message = self.node.ports["ccon_L"].rx_input()
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
                    yield self.node.qmemory.execute_program(self._program, qubit_mapping=[1]) #Llama a la función de abajo
                self.send_signal(Signals.SUCCESS)
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
        yield self.run() #Aquí se queda atascado

def setup_datacollector_r(network, protocol):
    # Ensure nodes are ordered in the chain:
    nodes = [network.nodes[name] for name in sorted(network.nodes.keys())]

    def calc_fidelity(evexpr):
        qubit_a, = nodes[0].qmemory.peek([0]) #cogemos el qubit guardado en la memoria del emisor
        qubit_b, = nodes[-1].qmemory.peek([1]) #cogemos el qubit guardado en la memoria del receptor
        global num #número de qubits calculados
        global rep #número de qubits que queremos calcular
        fidelity = ns.qubits.fidelity([qubit_a, qubit_b], ns.qubits.ketstates.b00, squared=True) #calculamos la fidelidad
        num += 1 #aumentamos l contador de qubits calculados
        if num >= rep: #si hemos calculado el número de qubits que queremos paramos la simulación
        	ns.sim_stop()
        return {"fidelity": fidelity} #devolvemos el resultado de la fidelidad

    
    dc = DataCollector(calc_fidelity, include_entity_name=False)
    dc.collect_on(py.EventExpression(source=protocol.subprotocols['CorrectProtocol'], event_type=Signals.SUCCESS.value))
    return dc
  
def repetidores(num_nodes=4, node_distance=20, num_iters=100, att=0.2, depol=0.07):
    ns.sim_reset()
    est_runtime = (0.5 + num_nodes - 1) * node_distance * 5e3
    network = setup_network(num_nodes, node_distance=node_distance, source_frequency=1e9 / est_runtime, att=att, depol=depol)
    protocol = setup_repeater_protocol(network)
    dc = setup_datacollector_r(network, protocol)
    protocol.start()
    ns.sim_run(est_runtime * num_iters * 10)
    global num
    print(f"num = {num}")
    return dc.dataframe

num = 0   

def create_plot(num_iters):
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    global rep
    global num
    rep = num_iters
    data = pd.DataFrame()
    for att in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    	if att == 0:
    		data[att] = directa(distance=25, att=att, depol=0.015)['fidelity']
    	else:
    		data[att] = data[0]
    data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
    data.plot(y='fidelity', yerr='sem', label="0 - Directa", ax=ax)
    #dist = [i for i in range(40, 60, 2)]
    for nodos in [3, 4, 5, 6, 7, 8]:
    	data = pd.DataFrame()
    	for att in [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    		data[att] = repetidores(num_nodes=nodos, node_distance=25/(nodos-1), num_iters=num_iters, att=att, depol=0.015*2)['fidelity']
    		num = 0
    	data = data.agg(['mean', 'sem']).T.rename(columns={'mean': 'fidelity'})
    	if nodos != 2:
    		data['fidelity'] = data['fidelity']*(2/3) + 1/3
    	data.plot(y='fidelity', yerr='sem', label=f"{nodos-2}", ax=ax)
    	print(f"Nodos: {nodos}")
    	print(f"{data}")
    	print("")
    plt.xlabel("atenuación")
    plt.ylabel("Fidelidad")
    plt.grid()
    plt.legend(title='número de repetidores:')
    plt.title("Comparación en función de la atenuación")
    plt.show()
	
if __name__ == "__main__":
	create_plot(500) 