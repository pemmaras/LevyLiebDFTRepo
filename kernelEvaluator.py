#This script is meant to scan over given input densities and for each density n, spit out LL[n] and the corresponding WF psi[n]
import sys

from sympy import false
sys.path.append('../')
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
        result_states = comm.bcast(result_states, root=min_rank)
        #~~~~~~~ End MPI stuff ~~~~~~~~
        self.wf_params = np.reshape(result_states,(result_states.shape[0]))
        print(f'n_i {n_i} and state {self.wf_params}')
        if rank == 0:
            print(f'for n_i {n_i}, returning energy {ene}')
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
        return scan_data
    
    def run_opt(self):
        eq_cons = {'type': 'eq', "fun": self.sum_cons }
        ineq_cons = {'type': 'ineq', "fun": self.pot_cons}
        n0 = np.copy(self.mf_ni)
        # res = basinhopping(self.dft_cost_fun, n0, niter=100, T=1.0, stepsize=0.5, minimizer_kwargs={'method':'SLSQP', 'bounds':((0.0,2.0),(0.0,2.0)), 'constraints':[eq_cons, ineq_cons]})
        res = minimize(self.dft_cost_fun, n0, method='SLSQP', tol=1e-7, bounds=((0.0,2.0),(0.0,2.0)), constraints=[eq_cons],options={'eps': 2e-3, 'maxiter': 1000, 'disp': True, 'finite_diff_rel_step':[0.1, 0.1]})
        print(res)
        return self.opt_data


### Set up a Kernel Matrix by explicit Levy-Lieb embedding
from Lattices import DimerLattice
from qiskit.opflow import StateFn, PauliExpectation
from qiskit import QuantumCircuit
from qiskit_nature.circuit.library import UCC
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge 

class kernelBuilder():
    def __init__(self, hubd, wf_anz):
        self.hubd = hubd
        self.wf_anz = wf_anz
        self.vqd = VQD([self.hubd.qTpU], self.wf_anz, 0, states=np.random.rand(len(self.wf_anz.preferred_init_points),1))

    def calc_rho_wf_map(self, num_train):
        self.latSolver = LatticeSolver(self.hubd)
        qmf_res = self.latSolver.quantum_solve([self.hubd.qmfH])
        self.mf_params = np.reshape(qmf_res[1],(qmf_res[1].shape[0]))
        
        comm = MPI.Comm(comm=MPI.COMM_WORLD)
        size = comm.Get_size()
        rank = comm.Get_rank()
        print(f'rank: {rank}, mf_params:{self.mf_params}')

        n_l = np.linspace(1.0, 0.01, num_train)

        res_data = []

        for rho in n_l:
            dloop = dft_looper(hubd=self.hubd, mf_params=self.mf_params, mf_ni=rho, wf_anz=self.wf_anz)
            res = dloop.run_scan(-0.0, 1)
            res_data.append(np.array(res))
        
        with open('lln_data_u'+str(self.hubd.U)+'.pic', 'wb') as fp:
            pickle.dump(res_data, fp)
            fp.close()        
    
    def read_saved_lln(self, fname):
        with open(fname, 'rb') as fp:
            u1_data = np.array(pickle.load(fp))
        self.xd = u1_data[:,[0,2]]
        self.yd = u1_data[:,1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.xd, self.yd, 
                                                        test_size=0.94, 
                                                        random_state=20)    #default rs=20
        print(self.X_train)

    def kernel_matrix(self, X_tt, X_tr):
        n_tr = X_tr.shape[0]
        n_tt = X_tt.shape[0]
        KM = np.eye(n_tt, n_tr)
        assert  X_tt.shape[1] == X_tr.shape[1]
        for i in range(n_tt):
            ni = X_tt[i,0]
            wfn_i_par = X_tt[i,1]['wfn']
            p_vec_i = list(self.vqd.wf_circ_op.parameters)[0].vector
            p_dict_i = {p_vec_i: wfn_i_par}
            for j in range(n_tr):
                nj = X_tr[j,0]
                wfn_j_par = X_tr[j,1]['wfn']
                p_vec_j = list(self.vqd.wf_circ_op.parameters)[0].vector
                p_dict_j = {p_vec_j: wfn_j_par}
                KM[i,j] = np.abs(self.vqd.get_overlap(self.wf_anz, p_dict_i, p_dict_j))
                print(f'ni={ni}, nj={nj}, KM[{i},{j}] = {KM[i,j]}')
        print(KM)
        return KM



init_qc = QuantumCircuit(4)
init_qc.x(0)
init_qc.x(3)
jwcon = QubitConverter(mapper=JordanWignerMapper(), two_qubit_reduction=False)
wf_anz = UCC(excitations='sd', qubit_converter=jwcon, num_particles=(1,1), 
                    num_spin_orbitals=4, reps=2, initial_state=init_qc, 
                    preserve_spin=False, generalized=True)
print(wf_anz)

### Construct lattice
num_nodes = 2
num_particles = 2
hopping_parameter =  0.5 
onsite_interaction = 1.0
vp = 0.0
potential = [vp, 0]
hubd = DimerLattice(num_particles, hopping_parameter, onsite_interaction, potential)

kb = kernelBuilder(hubd, wf_anz)
kb.read_saved_lln('lln_u1_data_v1.pic')

kernel_train = kb.kernel_matrix(kb.X_train,kb.X_train)

model = KernelRidge(alpha=0, kernel='precomputed')
model.fit(kernel_train, kb.y_train)

indices = np.random.choice(len(kb.X_test), len(kb.X_test), replace=False)
X_tt_s = kb.X_test[indices]
y_tt_s = kb.y_test[indices]
kernel_test = kb.kernel_matrix(X_tt_s, kb.X_train)

y_pred = model.predict(kernel_test)

result_data = np.hstack((X_tt_s[:,0].reshape(-1,1),y_tt_s.reshape(-1, 1),y_pred.reshape(-1, 1)))
print(result_data)
with open('lln_krr_fit_u'+str(hubd.U)+'_ntr2_R2.pic', 'wb') as fp:
    pickle.dump(result_data, fp)
    fp.close()    

import matplotlib.pyplot as plt
plt.scatter(y_tt_s, y_pred)
plt.grid()
plt.show()

# plt.scatter(kb.X_train[:,0], kb.y_train)
plt.plot(kb.xd[:,0],kb.yd)
plt.scatter(X_tt_s[:,0],y_tt_s)
plt.scatter(X_tt_s[:,0],y_pred)
plt.legend(['Full', 'True', 'Pred'])
plt.grid()
plt.show()