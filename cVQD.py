from typing import List
import numpy as np
from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization.electronic import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper, JordanWignerMapper
from qiskit_nature.properties.second_quantization.electronic import ElectronicEnergy, ParticleNumber, AngularMomentum, Magnetization, DipoleMoment
from qiskit_nature.transformers.second_quantization.electronic import ActiveSpaceTransformer
from qiskit_nature.algorithms.ground_state_solvers import GroundStateEigensolver

from qiskit_nature.circuit.library import UCCSD, HartreeFock, initial_states

from qiskit.algorithms.optimizers import SPSA
from qiskit_nature.operators.second_quantization import FermionicOp
from qiskit.opflow import SummedOp, StateFn, PrimitiveOp

from qiskit.opflow import StateFn, ExpectationFactory, CircuitStateFn, ListOp, CircuitSampler
from qiskit.opflow import I, X, Y, Z, Zero, One, Plus, Minus, PauliExpectation
from qiskit.providers.aer import AerSimulator
from qiskit.algorithms.optimizers import P_BFGS, L_BFGS_B, NELDER_MEAD, COBYLA
import numpy.random as npr
import scipy.linalg as sl
from scipy.optimize import minimize, basinhopping

class VQD:
    
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        self.qOps = qOps
        self.wfnA = wfnA
        self.nEx = nEx
        self.states = states
        
        hq_op = self.qOps[0]  #Hamiltonian
        
        #Construct the n-qubit |0><0| projection operator needed for overlap estimation using opflow .
        Qp = (1/2) * ( I + Z )
        self.zProj = Qp
        for iq in range(1, hq_op.num_qubits):
            self.zProj ^= Qp

        #Recast the wavefunction ansatz to a CircuitStateFn for opflow
        self.wf_circ_op = CircuitStateFn(wfnA)
                   
        self.backend = AerSimulator(device='CPU', blocking_enable=True, method='statevector', batched_shots_gpu=True)
        
        expectation = ExpectationFactory.build(operator=hq_op,backend=self.backend,include_custom=True)
        observable_meas = expectation.convert(StateFn(hq_op, is_measurement=True))
        self.expect_op = observable_meas.compose(self.wf_circ_op).reduce()
        
        exp_proj = ExpectationFactory.build(operator=self.zProj,backend=self.backend,include_custom=True)
        self.proj_meas = exp_proj.convert(StateFn(self.zProj, is_measurement=True))
        
        if not self.states.any():
            self.wfn_params = np.eye(len(wfnA.parameters), nEx+1)
        else:
            self.wfn_params = self.states
        self.nFound = 0
        self.LagM = 3.0*np.ones(nEx)
        # self.optimizer = P_BFGS(maxfun=2000, ftol=1e-8, iprint=-1, max_processes=1)
        self.optimizer = L_BFGS_B(maxfun=2000, iprint=-1)
        # self.optimizer = NELDER_MEAD(maxiter=2000, disp=False, tol=1e-6)
        self.fcount = 0
        self.curr_state = 0
        self.eigenvals = []

    def get_overlap(self, wfn_anz, p_dict1, p_dict2):
        """
        Function to calculate state overlap <state2 | state1> using OpFlow
        ----Parameters----
        wfn_anz: A parameterized circuit StateFn ansatz with unbound parameters
        p_dict1: A dictionary of paramers to bind to obtain state1
        p_dict2: A dictionary of paramers to bind to obtain state2
        """
        state1 = wfn_anz.bind_parameters(p_dict1)
        state2 = wfn_anz.bind_parameters(p_dict2)
        state = CircuitStateFn(state1.compose(state2.inverse()))
        exp_ovlp = self.proj_meas.compose(state).reduce()
        return exp_ovlp.eval()
    
    def get_expectation(self, Mobj, p_dict):
        """
        Function to calculate expectation values using OpFlow
        ----Parameters----
        Mobj: An OpFlow diagonal measurement object with unbound parameters
        p_dict: A dictionary of paramers to bind
        """
        #Bind all relevant parameters
        expects = Mobj.bind_parameters(p_dict)
        
        #Call the eval function on the bound expectation object to get the results
        res = expects.eval()
        return res    
    
    def cost_fun_gen(self, params):
        """
        Function that evalues the cost function for deflation
        ----Parameters-----
        params: parameters of the wf circuit ansatz as a list
        """
        # Create a params dict from the provided list
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}

        # Evaluate the Hamiltonian expectation value for the current parameter set using the saved diag measrement obj
        ev = self.get_expectation(self.expect_op, p_dict)
        pen = 0
        for i in range(self.nFound):
            params_i = (self.wfn_params[:,i])
            p_dict_i = {p_vec: params_i}
            ovlp = self.get_overlap(self.wfnA, p_dict, p_dict_i)
            pen += self.LagM[0] * np.abs(ovlp)
            
        cost = np.real(ev) +  pen
        
        self.fcount += 1
        # if self.fcount%50 == 0:
        #     print(f'cost, ev, pen is: {cost, ev, pen}')
        return cost
    
    def calculate(self):
        #start with the 0th exciated state i.e. ground state
        for iex in range(self.nEx+1):
            self.curr_state = iex
            if self.states.any():
                p0 = self.states[:,iex]
            else:
                if iex == 0:
                    p0 = self.wfnA.preferred_init_points
                else:
                    p0 = (np.pi/2) * np.random.rand(len(self.wfnA.preferred_init_points))
                
            bounds = [(-2 * np.pi, 2 * np.pi)] * len(self.wfnA.parameters)
            res = self.optimizer.minimize(fun=self.cost_fun_gen, x0=p0, jac=None, bounds=bounds)
            self.fcount = 0
            eigenval = res.fun
            eig_params = res.x
            self.nFound += 1
            if self.nEx >= iex:
                self.wfn_params[:,iex] = eig_params
            # print(f'index, eigenValue, eigenVector:{iex, eigenval, eig_params}')
            self.eigenvals.append(eigenval)
        return (self.eigenvals, self.wfn_params)

class ExcitedStateCalculator:
    def __init__(self, geometry):
        self.geometry = geometry
        # define molecule
        self.mol = Molecule(geometry=geometry, charge=0, multiplicity=1)

        # specify PYSCF classical driver
        self.driver = ElectronicStructureMoleculeDriver(self.mol, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)

        # run the classical SCF driver
        self.mf_result = self.driver.run()  
        
        # get particle number
        self.mf_energy = self.mf_result.get_property(ElectronicEnergy)
        self.particle_number = self.mf_result.get_property(ParticleNumber)
        self.num_electrons = self.particle_number.num_particles
        self.num_spin_orbitals = self.particle_number.num_spin_orbitals
        self.num_molecular_orbitals = int(self.num_spin_orbitals//2)
        
        #specify active space transformation (not needed for this example)
        active_space_trafo = ActiveSpaceTransformer(num_electrons=self.num_electrons, 
                                                    num_molecular_orbitals=self.num_molecular_orbitals)

        # define electronic structure problem
        self.es_problem = ElectronicStructureProblem(self.driver, transformers=[active_space_trafo])   
        
        #specify a quibit mapping (Jordan Wigner) of second quantized operators
        qubit_converter = QubitConverter(JordanWignerMapper(), two_qubit_reduction=False)
        
        #specify a HartreeFock initial state 
        hf_init = HartreeFock(self.num_spin_orbitals, self.num_electrons, qubit_converter)
        
        #instantiate a generalized UCCGSD ansatz circuit with 2 Trotter steps.
        self.wf_ansatz = UCCSD(qubit_converter=qubit_converter, num_particles=self.num_electrons, 
                               num_spin_orbitals=self.num_spin_orbitals, reps=2, initial_state=hf_init, 
                               preserve_spin=False, generalized=False)

        #perform the Fermion to qubit mapping of second quantized operators 
        self.q_ops = qubit_converter.map(self.es_problem.second_q_ops())
        
        
    def quantum_solve(self, num_excited_states, guess_state_parameters=np.array([])):
        print(self.wf_ansatz)
        vqdi = VQD(self.q_ops, self.wf_ansatz, num_excited_states, states=guess_state_parameters )
        (eigenvalues, result_states) = vqdi.calculate()
        #add nuclear repulsion to electronic energy
        eigenvalues = [ x + self.mf_energy.nuclear_repulsion_energy for x in eigenvalues ]
        return (eigenvalues, result_states)
    
    def classical_solve(self, num_excited_states):
        #Extract relevant qubit operators
        hq_op = self.q_ops[0]  #Hamiltonian
        nq_op = self.q_ops[1]  #Number operator
        
        #extract matrix representations of qubit operators
        H_Mat = hq_op.to_matrix()
        N_Mat = nq_op.to_matrix()
        
        #diagonalize to extrant eigen energies and eigenfunctions
        omega, psi = sl.eigh(H_Mat)
        
        #evaluate matrix elements of operators in eigenbasis
        N_expect = np.dot(psi.conj().T,np.dot(N_Mat, psi))
        E_expect = np.dot(psi.conj().T,np.dot(H_Mat, psi))
        
        #collect energies corresponding to states with the correct number of electrons
        c_eigs = []
        for i in range(N_Mat.shape[0]):
            if abs(N_expect[i,i] - np.sum(self.num_electrons)) < 1e-8:
                c_eigs.append(E_expect[i,i] + self.mf_energy.nuclear_repulsion_energy)
        c_eigs = np.real(np.array(c_eigs))
        return np.sort(c_eigs)[0:num_excited_states+1]



class SpinConstrainedVQD(VQD):
    
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        super().__init__(qOps, wfnA, nEx, states)
        
        # To implement explicit total Spin constraint use the S^2 operator 
        #from the set of available qubit operators.
        S2_op = self.qOps[2]
        exp_S2 = ExpectationFactory.build(operator=S2_op,backend=self.backend,include_custom=True)
        S2_meas = exp_S2.convert(StateFn(S2_op, is_measurement=True))
        self.expect_S2 = S2_meas.compose(self.wf_circ_op).reduce()
            
    def cost_fun_gen(self, params):
        """
        Function that evalues the cost function for deflation
        ----Parameters-----
        params: parameters of the wf circuit ansatz as a list
        """
        # Create a params dict from the provided list
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}

        # Evaluate the Hamiltonian expectation value for the current parameter set using the saved diag measrement obj
        ev = self.get_expectation(self.expect_op, p_dict)
        
        # Evaluate the S^2 expectation value for total spin constraint
        S2_ev = self.get_expectation(self.expect_S2, p_dict)

        pen = 0
        for i in range(self.nFound):
            params_i = (self.wfn_params[:,i])
            p_dict_i = {p_vec: params_i}
            ovlp = self.get_overlap(self.wfnA, p_dict, p_dict_i)
            pen += self.LagM[0] * np.abs(ovlp)
            
        cost = np.real(ev) +  pen + 3 * np.real(S2_ev) #cost now includes additional penalty to force total spin towards 0.
        
        self.fcount += 1
        if self.fcount%50 == 0:
            print(f'cost, ev, pen, S2_ev is: {cost, ev, pen, S2_ev}')
        return cost        

class ConstrainedExcitedStateCalculator(ExcitedStateCalculator):
    def __init__(self, geometry):
        super().__init__(geometry)        
        
    def quantum_solve(self, num_excited_states, guess_state_parameters=None):
        print(self.wf_ansatz)
        vqdi = SpinConstrainedVQD(self.q_ops, self.wf_ansatz, num_excited_states, states=guess_state_parameters )
        (eigenvalues, result_states) = vqdi.calculate()
        #add nuclear repulsion to electronic energy
        eigenvalues = [ x + self.mf_energy.nuclear_repulsion_energy for x in eigenvalues ]
        return (eigenvalues, result_states)
    
    def classical_solve(self, num_excited_states):
        #Extract relevant qubit operators
        hq_op = self.q_ops[0]  #Hamiltonian
        nq_op = self.q_ops[1]  #Number operator
        s2_op = self.q_ops[2]  #S^2 operator
        
        #extract matrix representations of qubit operators
        H_Mat = hq_op.to_matrix()
        N_Mat = nq_op.to_matrix()
        S2_Mat = s2_op.to_matrix()
        
        #diagonalize to extrant eigen energies and eigenfunctions
        omega, psi = sl.eigh(H_Mat)
        
        #evaluate matrix elements of operators in eigenbasis
        N_expect = np.dot(psi.conj().T,np.dot(N_Mat, psi))
        E_expect = np.dot(psi.conj().T,np.dot(H_Mat, psi))
        S2_expect = np.dot(psi.conj().T,np.dot(S2_Mat, psi))
        
        #collect energies corresponding to states with the correct number of electrons and total Spin
        c_eigs = []
        for i in range(N_Mat.shape[0]):
            if abs(N_expect[i,i] - np.sum(self.num_electrons)) < 1e-8 and abs(S2_expect[i,i]) < 1e-2:
                c_eigs.append(E_expect[i,i] + self.mf_energy.nuclear_repulsion_energy)
        c_eigs = np.real(np.array(c_eigs))
        return np.sort(c_eigs)[0:num_excited_states+1]

class DensityConstrainedVQD(VQD):
    
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        super().__init__(qOps, wfnA, nEx, states)
        
        self.optimizer = L_BFGS_B(maxfun=500, iprint=-1)

        # To implement explicit site occupation constraints we need a list of site occupation operators
        qns_op = self.qOps[2]
        exp_qnp = [ ExpectationFactory.build(operator=qns_op[i],backend=self.backend,include_custom=True) for i in range(len(qns_op))  ]
        qnp_meas = [ exp_qnp[i].convert(StateFn(qns_op[i], is_measurement=True)) for i in range(len(qns_op)) ]
        self.expect_qnp = [ opmeas.compose(self.wf_circ_op).reduce() for opmeas in qnp_meas ]
            
    def cost_fun_gen(self, params):
        """
        Function that evalues the cost function for deflation
        ----Parameters-----
        params: parameters of the wf circuit ansatz as a list
        """
        # Create a params dict from the provided list
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}

        # Evaluate the Hamiltonian expectation value for the current parameter set using the saved diag measrement obj
        ev = self.get_expectation(self.expect_op, p_dict)
        
        # Evaluate the expectation values of  zce2            the list of site occupation ops
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = 1.0e8 * np.real(np.dot(np.conj(qnp_ev),qnp_ev))

        pen = 0
        for i in range(self.nFound):
            params_i = (self.wfn_params[:,i])
            p_dict_i = {p_vec: params_i}
            ovlp = self.get_overlap(self.wfnA, p_dict, p_dict_i)
            pen += self.LagM[0] * np.abs(ovlp)
            
        cost = np.real(ev) +  pen + dnpen #cost now includes additional penalty to force total spin towards 0.
        if self.fcount%100 == 0:
            print(f'DCSearch: fcount, cost, ev, pen, qnp_ev is: {self.fcount, cost, ev, pen, dnpen}')        
        self.fcount += 1
        return cost  

class LLConstrainedVQD(VQD):
    
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        super().__init__(qOps, wfnA, nEx, states)
        
        # To implement explicit site occupation constraints we need a list of site occupation operators
        qns_op = self.qOps[2]
        exp_qnp = [ ExpectationFactory.build(operator=qns_op[i],backend=self.backend,include_custom=True) for i in range(len(qns_op))  ]
        qnp_meas = [ exp_qnp[i].convert(StateFn(qns_op[i], is_measurement=True)) for i in range(len(qns_op)) ]
        self.expect_qnp = [ opmeas.compose(self.wf_circ_op).reduce() for opmeas in qnp_meas ]
            
    def cost_fun_gen(self, params):
        """
        Function that evalues the cost function for deflation
        ----Parameters-----
        params: parameters of the wf circuit ansatz as a list
        """
        # Create a params dict from the provided list
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}

        # Evaluate the Hamiltonian expectation value for the current parameter set using the saved diag measrement obj
        ev = self.get_expectation(self.expect_op, p_dict)
        
        # Evaluate the expectation values of the list of site occupation ops
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = np.real(np.dot(np.conj(qnp_ev),qnp_ev))

        pen = 0
        for i in range(self.nFound):
            params_i = (self.wfn_params[:,i])
            p_dict_i = {p_vec: params_i}
            ovlp = self.get_overlap(self.wfnA, p_dict, p_dict_i)
            pen += self.LagM[0] * np.abs(ovlp)
            
        cost = np.real(ev) +  pen  #+ dnpen #cost now includes additional penalty to force total spin towards 0.
        if self.fcount%50 == 0:
            print(f'LLSearch: fcount, cost, ev, pen, qnp_ev is: {self.fcount, cost, ev, pen, dnpen}')        
        self.fcount += 1
        return cost  

    def dens_cons(self, params):
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = np.real(np.dot(np.conj(qnp_ev),qnp_ev))
        return dnpen
        # return np.array(qnp_ev)
    

    def calculate(self):
        my_constraints = ({'type': 'eq', "fun": self.dens_cons })
        #start with the 0th exciated state i.e. ground state
        for iex in range(self.nEx+1):
            self.curr_state = iex
            if self.states.any():
                p0 = self.states[:,iex]
            else:
                if iex == 0:
                    p0 = self.wfnA.preferred_init_points
                else:
                    p0 = (np.pi/2) * np.random.rand(len(self.wfnA.preferred_init_points))
                
            bounds = [(-2 * np.pi, 2 * np.pi)] * len(self.wfnA.parameters)
            res = minimize(self.cost_fun_gen, p0, method='SLSQP',bounds=bounds, constraints=my_constraints ,options={'maxiter': 1000, 'disp': True })
            self.fcount = 0
            eigenval = res.fun
            eig_params = res.x
            self.nFound += 1
            if self.nEx >= iex:
                self.wfn_params[:,iex] = eig_params
            print(f'index, eigenValue, eigenVector:{iex, eigenval, eig_params}')
            self.eigenvals.append(eigenval)
        return (self.eigenvals, self.wfn_params)


class DensityConstrainer(VQD):
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        super().__init__(qOps, wfnA, nEx, states)
        
        # To implement explicit site occupation constraints we need a list of site occupation operators
        qns_op = self.qOps[2]
        exp_qnp = [ ExpectationFactory.build(operator=qns_op[i],backend=self.backend,include_custom=True) for i in range(len(qns_op))  ]
        qnp_meas = [ exp_qnp[i].convert(StateFn(qns_op[i], is_measurement=True)) for i in range(len(qns_op)) ]
        self.expect_qnp = [ opmeas.compose(self.wf_circ_op).reduce() for opmeas in qnp_meas ]

    def dens_cons(self, params):
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = np.real(np.dot(np.conj(qnp_ev),qnp_ev))
        return dnpen
        # return np.array(qnp_ev)
    
    def calculate(self):
        #start with the 0th exciated state i.e. ground state
        for iex in range(self.nEx+1):
            self.curr_state = iex
            if self.states.any():
                p0 = self.states[:,iex]
            else:
                if iex == 0:
                    p0 = self.wfnA.preferred_init_points
                else:
                    p0 = (np.pi/2) * np.random.rand(len(self.wfnA.preferred_init_points))
                
            bounds = [(-2 * np.pi, 2 * np.pi)] * len(self.wfnA.parameters)
            # res = basinhopping(self.cost_fun_gen, p0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs={'method':'SLSQP', 'bounds':bounds, 'constraints':my_constraints})
            res = minimize(self.dens_cons, p0, method='L-BFGS-B')
            # res = self.optimizer.minimize(fun=self.cost_fun_gen, x0=p0, jac=None, bounds=bounds)
            eigenval = res.fun
            eig_params = res.x
            self.nFound += 1
            if self.nEx >= iex:
                self.wfn_params[:,iex] = eig_params
            # print(f'Density Constrainer: index, minVal, solutionVector:{iex, eigenval, eig_params}')
            self.eigenvals.append(eigenval)
        return (self.eigenvals, self.wfn_params)

class COBYLALLConstrainedVQD(VQD):
    
    def __init__(self, qOps, wfnA, nEx, states=np.array([])):
        super().__init__(qOps, wfnA, nEx, states)
        
        # To implement explicit site occupation constraints we need a list of site occupation operators
        qns_op = self.qOps[2]
        exp_qnp = [ ExpectationFactory.build(operator=qns_op[i],backend=self.backend,include_custom=True) for i in range(len(qns_op))  ]
        qnp_meas = [ exp_qnp[i].convert(StateFn(qns_op[i], is_measurement=True)) for i in range(len(qns_op)) ]
        self.expect_qnp = [ opmeas.compose(self.wf_circ_op).reduce() for opmeas in qnp_meas ]
            
    def cost_fun_gen(self, params):
        """
        Function that evalues the cost function for deflation
        ----Parameters-----
        params: parameters of the wf circuit ansatz as a list
        """
        # Create a params dict from the provided list
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}

        if self.fcount == 0:
            print(f'new params: {params}')

        # Evaluate the Hamiltonian expectation value for the current parameter set using the saved diag measrement obj
        ev = self.get_expectation(self.expect_op, p_dict)
        
        # Evaluate the expectation values of the list of site occupation ops
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = np.real(np.dot(np.conj(qnp_ev),qnp_ev))

        pen = 0
        for i in range(self.nFound):
            params_i = (self.wfn_params[:,i])
            p_dict_i = {p_vec: params_i}
            ovlp = self.get_overlap(self.wfnA, p_dict, p_dict_i)
            pen += self.LagM[0] * np.abs(ovlp)
            
        cost = np.real(ev) +  pen  #+ dnpen #cost now includes additional penalty to force total spin towards 0.
        if self.fcount%50 == 0:
            print(f'cost, ev, pen, dnpen is: {cost, ev, pen, dnpen}')
        self.fcount += 1
        return cost  

    def dens_cons(self, params):
        p_vec = list(self.wf_circ_op.parameters)[0].vector
        p_dict = {p_vec: params}
        qnp_ev =  [ self.get_expectation(qnp_exp, p_dict) for qnp_exp in self.expect_qnp ]
        dnpen = 1e-8 - np.real(np.dot(np.conj(qnp_ev),qnp_ev))
        return dnpen
        # return np.array(qnp_ev)
    

    def calculate(self):
        my_constraints = ({'type': 'ineq', "fun": self.dens_cons })
        #start with the 0th exciated state i.e. ground state
        for iex in range(self.nEx+1):
            self.curr_state = iex
            if self.states.any():
                p0 = self.states[:,iex]
            else:
                if iex == 0:
                    p0 = self.wfnA.preferred_init_points
                else:
                    p0 = (np.pi/2) * np.random.rand(len(self.wfnA.preferred_init_points))
                
            bounds = [(-2 * np.pi, 2 * np.pi)] * len(self.wfnA.parameters)
            # res = basinhopping(self.cost_fun_gen, p0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs={'method':'SLSQP', 'bounds':bounds, 'constraints':my_constraints})
            res = minimize(self.cost_fun_gen, p0, method='COBYLA',bounds=bounds, constraints=my_constraints ,options={'maxiter': 1000, 'disp': True })
            self.fcount = 0
            # res = self.optimizer.minimize(fun=self.cost_fun_gen, x0=p0, jac=None, bounds=bounds)
            eigenval = res.fun
            eig_params = res.x
            self.nFound += 1
            if self.nEx >= iex:
                self.wfn_params[:,iex] = eig_params
            print(f'index, eigenValue, eigenVector:{iex, eigenval, eig_params}')
            self.eigenvals.append(eigenval)
        return (self.eigenvals, self.wfn_params)



# from Lattices import DimerLattice, Vector
# from qiskit import QuantumCircuit
# from qiskit_nature.circuit.library import UCC

# class LatticeSolver():
#     def __init__(self, lattice: DimerLattice ) -> None:
#         #Initialize wave function ansatz
#         self.lat = lattice
#         self.nqubits = lattice.nsites * 2
#         init_qc = QuantumCircuit(self.nqubits)
#         print(lattice.num_particles)
#         n_up = int(np.ceil(lattice.num_particles/2))
#         # n_dn = int(np.floor(lattice.num_particles/2))
#         print(type(n_up))
#         # exit()
#         # assert self.nqubits >= n_up + n_dn 
#         # for ie in range(n_up):
#         #     iq = 2*ie
#         #     init_qc.x(iq)
#         # for ie in range(n_dn):
#         #     iq = self.nqubits - (2*ie + 1)
#         #     init_qc.x(iq)
#         # #instantiate a generalized UCCGSD ansatz circuit with 2 Trotter steps.
#         # self.wf_anz = UCC(excitations='sd', qubit_converter=lattice.jw_converter, num_particles=(n_up,n_dn), 
#         #                num_spin_orbitals=self.nqubits, reps=2, initial_state=init_qc, 
#         #                preserve_spin=False, generalized=True)

#     def classical_solve(self, Ops: Vector) -> List:
#         #Ops [ Hamiltonian, ParticleNumber, ...]
#         assert len(Ops) >= 2  

#         HM = Ops[0].to_matrix()
#         #Diagonalize
#         w, Psi = sl.eig(HM)
#         #calculate matrix elements for number operator to select relevant state
#         NM = Ops[1].to_matrix()
#         N_ev = np.dot(Psi.conj().T,np.dot(NM, Psi))
#         expect_vals=[]
#         for j in range(len(w)):
#             j_row = [j]
#             if np.abs(N_ev[j,j] - self.lat.num_particles) < 0.1:
#                 psi_j = Psi[:,j]
#                 for qop in Ops:
#                     if type(qop) == list:
#                         qop_ev = [ np.dot(psi_j.conj().T,np.dot(op.to_matrix(), psi_j)) for op in qop ]
#                     else:
#                         qop_ev = np.dot(psi_j.conj().T,np.dot(qop.to_matrix(), psi_j))
#                     j_row.append(qop_ev)
#                 expect_vals.append(j_row)
#         return expect_vals


#     # def mean_field_qsolve(self):
#     #     mfvqd = VQD([self.lat.qmfH], self.wf_anz, 0, states=np.random.rand(len(self.wf_anz.preferred_init_points),1))
#     #     (eigenvalues, result_states) = mfvqd.calculate()

