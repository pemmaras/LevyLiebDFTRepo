from math import pi
import numpy as np
import retworkx as rx
import scipy.linalg as sl
from qiskit_nature.problems.second_quantization.lattice import (
    BoundaryCondition,
    Lattice,
    LatticeDrawStyle,
    LineLattice, 
    FermiHubbardModel
)
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
from typing import List


# Define some Type Aliases
Vector = List[float]

class DimerLattice:
  # instance attributes
  def __init__(self, num_particles: float, hopping_parameter: float, onsite_interaction: float, vext: Vector ) -> None:
    self.nsites = 2
    self.num_particles = num_particles
    self.t = hopping_parameter
    self.U = onsite_interaction
    self.vext = vext
    self.jw_converter = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
    self.setupOps()
    self.setupQubitOps()

  # instance method
  def setupOps(self):
    self.Hop   = self.getFHMOp(self.t, self.U, self.vext).reduce()
    self.Top   = self.getFHMOp(self.t, 0, np.zeros(len(self.vext))).reduce()
    self.TpUop = self.getFHMOp(self.t, self.U, np.zeros(len(self.vext))).reduce()
    self.Uop   = self.getFHMOp(0, self.U, np.zeros(len(self.vext))).reduce()
    self.Vop   = self.getFHMOp(0, 0, self.vext).reduce()
    self.mfHop = self.getFHMOp(self.t, 0, self.vext).reduce()
    #Set up number operator for Hubbard dimer
    self.N_op = sum(FermionicOp("+_"+str(i)+" -_"+str(i), register_length=4, display_format='dense') for i in range(2*self.nsites))
    #Set up spin-resolved site occupation
    self.nsigma = [ FermionicOp("+_"+str(i)+" -_"+str(i), register_length=4, display_format='dense') for i in range(2*self.nsites) ]
    #Set up total-site occupation op
    self.nsite = [ FermionicOp("+_"+str(2*i)+" -_"+str(2*i), register_length=4, display_format='dense') + 
                      FermionicOp("+_"+str(2*i+1)+" -_"+str(2*i+1), register_length=4, display_format='dense') for i in range(self.nsites) ]
    #Set up the spin-density operator
    self.mz = [ FermionicOp("+_"+str(2*i)+" -_"+str(2*i), register_length=4, display_format='dense') + 
                  (-1.0)*FermionicOp("+_"+str(2*i+1)+" -_"+str(2*i+1), register_length=4, display_format='dense') for i in range(self.nsites) ]

  #instance method
  def setupQubitOps(self):
    self.qH   = self.jw_converter.convert(self.Hop, num_particles=self.num_particles)
    self.qT   = self.jw_converter.convert(self.Top, num_particles=self.num_particles)
    self.qTpU = self.jw_converter.convert(self.TpUop, num_particles=self.num_particles)
    self.qU   = self.jw_converter.convert(self.Uop, num_particles=self.num_particles)
    self.qV   = self.jw_converter.convert(self.Vop, num_particles=self.num_particles)
    self.qmfH = self.jw_converter.convert(self.mfHop, num_particles=self.num_particles)
    self.qN   = self.jw_converter.convert(self.N_op, num_particles=self.num_particles)
    self.qnsigma   = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.nsigma ]
    self.qnsite = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.nsite ]
    self.qmz  = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in self.mz ]

  #instance method
  def densityPenaltyOp(self, ni: Vector):
    #Construct the operator list [ n_i_op - ni ]
    dp_op = [ self.nsite[i] + (-1.0 * ni[i])*FermionicOp.one(4) for i in range(len(ni) - 1) ]
    qdp_op = [ self.jw_converter.convert(op, num_particles=self.num_particles) for op in dp_op ]
    return qdp_op 

  # instance method
  def getFHMOp(self, t: float, U: float, vext: Vector):
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(self.nsites))
    weighted_edge_list = [
        (0, 0, vext[0]),
        (1, 1, vext[1]),
        (0, 1, t),
    ]
    graph.add_edges_from(weighted_edge_list)
    general_lattice = Lattice(graph)
    set(general_lattice.graph.weighted_edge_list())
    fhm = FermiHubbardModel(general_lattice, onsite_interaction=U)
    return fhm.second_q_ops()
