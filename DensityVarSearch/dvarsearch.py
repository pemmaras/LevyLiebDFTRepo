#This script is meant to scan over given input densities and for each density n, spit out LL[n] and the corresponding WF psi[n]
import sys
sys.path.append('../')
sys.path.append('../../')
from logging import root
from math import pi
import numpy as np
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
from Lattices import DimerLattice, Vector
from qiskit import QuantumCircuit
from qiskit_nature.circuit.library import UCC
from cVQD import VQD, DensityConstrainedVQD, LLConstrainedVQD

from mpi4py import MPI

class LatticeSolver():
    def __init__(self, lattice: DimerLattice ) -> None:
        #Initialize wave function ansatz
        self.lat = lattice
        self.nqubits = lattice.nsites * 2
        init_qc = QuantumCircuit(self.nqubits)
        n_up = int(np.ceil(lattice.num_particles/2))
        n_dn = int(np.floor(lattice.num_particles/2))
        assert self.nqubits >= n_up + n_dn 
        for ie in range(n_up):
            iq = 2*ie
            init_qc.x(iq)
        for ie in range(n_dn):
            iq = self.nqubits - (2*ie + 1)
            init_qc.x(iq)
        #instantiate a generalized UCCGSD ansatz circuit with 2 Trotter steps.
        self.wf_anz = UCC(excitations='sd', qubit_converter=lattice.jw_converter, num_particles=(n_up,n_dn), 
                       num_spin_orbitals=self.nqubits, reps=2, initial_state=init_qc, 
                       preserve_spin=False, generalized=True)
    
    def classical_solve(self, Ops: Vector) -> List:
        #Ops [ Hamiltonian, ParticleNumber, ...]
        assert len(Ops) >= 2  

        HM = Ops[0].to_matrix()
        #Diagonalize
        w, Psi = sl.eig(HM)
        #calculate matrix elements for number operator to select relevant state
        NM = Ops[1].to_matrix()
        N_ev = np.dot(Psi.conj().T,np.dot(NM, Psi))
        expect_vals=[]
        for j in np.argsort(w):
            j_row = [j]
            if np.abs(N_ev[j,j] - self.lat.num_particles) < 0.1:
                psi_j = Psi[:,j]
                for qop in Ops:
                    if type(qop) == list:
                        qop_ev = [ np.dot(psi_j.conj().T,np.dot(op.to_matrix(), psi_j)) for op in qop ]
                    else:
                        qop_ev = np.dot(psi_j.conj().T,np.dot(qop.to_matrix(), psi_j))
                    j_row.append(np.real(qop_ev))
                expect_vals.append(j_row)
        return expect_vals


    def quantum_solve(self, Ops: Vector) -> List:
        vqd = VQD([Ops[0]], self.wf_anz, 0, states=np.random.rand(len(self.wf_anz.preferred_init_points),1))
        (eigenvalues, result_states) = vqd.calculate()
        return (eigenvalues, result_states)


from cVQD import LLConstrainedVQD, DensityConstrainer, COBYLALLConstrainedVQD
from scipy.optimize import minimize, basinhopping
class dft_looper():
    def __init__(self, hubd, mf_params, mf_ni, wf_anz):
        self.hubd = hubd
        self.wf_params = mf_params
        self.mf_ni = np.real(mf_ni)
        self.wf_anz = wf_anz
        self.mf_pot = np.dot(self.hubd.vext, mf_ni)
        self.opt_data = []


    def dft_cost_fun(self, n_i):
        comm = MPI.Comm(comm=MPI.COMM_WORLD)
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f'new density: {n_i}')
        
        qdpn = self.hubd.densityPenaltyOp(n_i)
        rand_dev = ( np.random.rand(len(self.wf_params),1) - 0.5 ) * (rank) * 0.01/size * np.pi 
        gstate = np.reshape(self.wf_params, (len(self.wf_params),1)) + rand_dev

        dcnstr = DensityConstrainer([self.hubd.qN, self.hubd.qN, qdpn], self.wf_anz, 0, states=gstate)
        (minvals, solStates) = dcnstr.calculate()
        gstate = np.reshape(solStates,(solStates.shape[0],1))

        vqdi = LLConstrainedVQD([self.hubd.qTpU, self.hubd.qN, qdpn], self.wf_anz, 0, states=gstate)
        (eigenvalues, result_states) = vqdi.calculate()
        tpu = np.real(eigenvalues[0])
        pot = np.dot(self.hubd.vext, n_i)
        ene = tpu + pot 
        #~~~~~~~ Begin MPI stuff ~~~~~~~~
        recv_buf = np.zeros(size)
        comm.barrier()
        comm.Allgather(np.array([ene]), recv_buf)
        # print(f'dcf rank:{rank}, recvbuf: {recv_buf}')
        min_rank = np.argmin(recv_buf)
        ene = np.min(recv_buf)
        # print(f'dcf rank: {rank}, minrank:{min_rank}')
        result_states = comm.bcast(result_states, root=min_rank)
        # comm.barrier()
        # print(f'dcf rank: {rank}, result_states:{result_states}')
        #~~~~~~~ End MPI stuff ~~~~~~~~
        self.wf_params = np.reshape(result_states,(result_states.shape[0]))
        # print(f'rank {rank} n_i {n_i} and state {self.wf_params}')
        # print(f'dcf RANK:{rank}, n_i: {n_i}, ene: {ene}, tpu:{tpu}, pot{pot}')
        if rank == 0:
            print(f'for n_i {n_i}, returning energy {ene}')
        # quit()
        self.opt_data.append([n_i[0], ene, {'wfn': self.wf_params}])
        return ene

    def sum_cons(self, n_i):
        total = self.hubd.num_particles - np.sum(n_i)
        return total

    def pot_cons(self, n_i):
        diff = np.dot(self.hubd.vext, n_i) - self.mf_pot
        return diff

    def run_scan(self, step_size, nstep):
        scan_data=[]
        for istep in range(nstep):
            n_i = np.copy(self.mf_ni) + np.array([step_size * istep, -step_size * istep])
            ene = self.dft_cost_fun(n_i)
            scan_data.append(self.opt_data[-1])            
            # scan_data.append([n_i[0],ene])
        return scan_data
    
    def run_opt(self):
        eq_cons = {'type': 'eq', "fun": self.sum_cons }
        ineq_cons = {'type': 'ineq', "fun": self.pot_cons}
        n0 = np.copy(self.mf_ni)
        # res = basinhopping(self.dft_cost_fun, n0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs={'method':'SLSQP', 'bounds':((0.0,2.0),(0.0,2.0)), 'constraints':[eq_cons, ineq_cons]})
        res = minimize(self.dft_cost_fun, n0, method='SLSQP', tol=1e-7, bounds=((0.0,2.0),(0.0,2.0)), constraints=[eq_cons],options={'eps': 2e-3, 'maxiter': 1000, 'disp': True, 'finite_diff_rel_step':[0.1, 0.1]})
        print(res)
        return res, self.opt_data


### Set up loop over potentials
from Lattices import DimerLattice
from qiskit.opflow import StateFn, PauliExpectation
from qiskit import QuantumCircuit
from qiskit_nature.circuit.library import UCC
import pickle
init_qc = QuantumCircuit(4)
init_qc.x(0)
init_qc.x(3)
jwcon = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
wf_anz = UCC(excitations='sd', qubit_converter=jwcon, num_particles=(1,1), 
                    num_spin_orbitals=4, reps=2, initial_state=init_qc, 
                    preserve_spin=False, generalized=True)
print(wf_anz)
vx=[]
fciy=[]
lly=[]
for hu in [1.0]: #np.linspace(1.5, 4.5, 7):
    ### Construct lattice
    num_nodes = 2
    num_particles = 2
    hopping_parameter =  0.5 
    onsite_interaction = hu
    vp = 1.0
    potential = [-vp/2, vp/2]
    vx.append(vp)
    hubd = DimerLattice(num_particles, hopping_parameter, onsite_interaction, potential)

    latSolver = LatticeSolver(hubd)

    ### FCI solution
    fci_res = latSolver.quantum_solve([hubd.qH])
    qfci_params = np.reshape(fci_res[1],(fci_res[1].shape[0]))
    fci_wfn = latSolver.wf_anz.bind_parameters(qfci_params)
    
    cesw = StateFn(fci_wfn)
    n_fci=[]
    for ni_op in hubd.qnsite:
        ni_meas = StateFn(ni_op).adjoint()
        diag_ni = ni_meas @ cesw
        diag_ni_expt = PauliExpectation().convert(diag_ni)
        n_fci.append(diag_ni_expt.eval())
    n_fci = np.real(n_fci)
    qTpU = hubd.qTpU
    tpu_meas = StateFn(qTpU).adjoint()
    diag_tpu = tpu_meas @ cesw
    diag_tpu_expt = PauliExpectation().convert(diag_tpu)
    tpu_fci = np.real(diag_tpu_expt.eval())
    etot_fci = np.real(fci_res[0][0])
    print(f'density: {n_fci} , tpu_fci: {tpu_fci}, ene_fci {etot_fci}, pot {etot_fci - tpu_fci}')
    fciy.append(tpu_fci)
    fci_result={ 'n_fci': n_fci, 'etot_fci':etot_fci, 'tpu_fci':tpu_fci, 'qfci_params':qfci_params }

    ###mf solution
    qmf_res = latSolver.quantum_solve([hubd.qmfH])
    mf_params = np.reshape(qmf_res[1],(qmf_res[1].shape[0]))
    mf_wfn = latSolver.wf_anz.bind_parameters(mf_params)

    #Set starting density to mf
    cesw = StateFn(mf_wfn)
    n_mf=[]
    for ni_op in hubd.qnsite:
        ni_meas = StateFn(ni_op).adjoint()
        diag_ni = ni_meas @ cesw
        diag_ni_expt = PauliExpectation().convert(diag_ni)
        n_mf.append(diag_ni_expt.eval())
    n_mf = np.real(n_mf)
    print(f'pot_mf: {np.dot(n_mf,potential)} , pot_fci: {np.dot(n_fci,potential)}')

    e_mf = np.array(qmf_res[0])
    comm = MPI.Comm(comm=MPI.COMM_WORLD)
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(f'{rank}, emf: {e_mf}')
    print(f'rank: {rank}, n_mf:{n_mf}')
    print(f'rank: {rank}, mf_params:{mf_params}')
    comm.barrier()
    dloop = dft_looper(hubd=hubd, mf_params=mf_params, mf_ni=n_mf, wf_anz=wf_anz)
    # data_res = dloop.run_scan(-0.01, 101)
    # dloop = dft_looper(hubd=hubd, mf_params=mf_params, mf_ni=n_mf, wf_anz=wf_anz)
    res, opt_data = dloop.run_opt()
    opt_data = np.array(opt_data)
    dvs_result={'res':res, 'opt_data':opt_data}
    
    full_result = {'fci': fci_result, 'dvs':dvs_result}
    
    print(f'=====Final result for U={hu}======')
    print(full_result)
    if rank == 0:
        with open('dvs_U'+str(hu)+'_v'+str(vp)+'.pic', 'wb') as fp:
            pickle.dump(full_result, fp)
        fp.close()    



