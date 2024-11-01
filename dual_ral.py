import hashing
from hashing import Hashing
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from qutip import (Options, basis, destroy, mesolve, sesolve, ptrace, qeye, tensor, wigner, states, displace, expect, coherent, fock, sigmax, sigmay,sigmaz, sigmam, Qobj, fidelity, propagator, fock_dm)
import qutip as qt
from qutip.qip.operations import (rx, ry, rz)

from typing import Optional, Tuple
import numpy as np
from numpy import ndarray
from scipy.special import comb


class Hashing:
    """Helper class for efficiently constructing raising and lowering operators
    using a global excitation cutoff scheme, as opposed to the more commonly used
    number of excitations per mode cutoff, which can be easily constructed
    using kronecker product. The ideas herein are based on the excellent
    paper
    [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010).
    """

    num_exc: int  # up to and including the number of global excitations to keep
    number_degrees_freedom: int  # number of degrees of freedom of the system

    def __init__(self, num_exc, number_degrees_freedom) -> None:
        self.num_exc = num_exc
        self.number_degrees_freedom = number_degrees_freedom
        self.sqrt_prime_list = np.sqrt(
            [
                2,
                3,
                5,
                7,
                11,
                13,
                17,
                19,
                23,
                29,
                31,
                37,
                41,
                43,
                47,
                53,
                59,
                61,
                67,
                71,
                73,
                79,
                83,
                89,
                97,
                101,
                103,
                107,
                109,
                113,
                127,
                131,
                137,
                139,
                149,
                151,
                157,
                163,
                167,
                173,
                179,
                181,
                191,
                193,
                197,
                199,
                211,
                223,
                227,
                229,
                233,
                239,
                241,
                251,
                257,
                263,
                269,
                271,
                277,
                281,
                283,
                293,
                307,
                311,
                313,
                317,
                331,
                337,
                347,
                349,
                353,
                359,
                367,
                373,
                379,
                383,
                389,
                397,
                401,
                409,
                419,
                421,
                431,
                433,
                439,
                443,
                449,
                457,
                461,
                463,
                467,
                479,
                487,
                491,
                499,
                503,
                509,
                521,
                523,
                541,
                547,
                557,
                563,
                569,
                571,
                577,
                587,
                593,
                599,
                601,
                607,
                613,
                617,
                619,
                631,
                641,
                643,
                647,
                653,
                659,
                661,
                673,
                677,
                683,
                691,
                701,
                709,
                719,
                727,
                733,
                739,
                743,
                751,
                757,
                761,
                769,
                773,
                787,
                797,
                809,
                811,
                821,
                823,
                827,
                829,
                839,
                853,
                857,
                859,
                863,
                877,
                881,
                883,
                887,
                907,
                911,
                919,
                929,
                937,
                941,
                947,
                953,
                967,
                971,
                977,
                983,
                991,
                997,
            ]
        )

    def gen_basis_vectors(self) -> ndarray:
        """Generate basis vectors using Zhang algorithm. `func` allows for inclusion of other vectors,
        such as those with negative entries (see CurrentMirrorGlobal)"""
        vector_list = [np.zeros(self.number_degrees_freedom)]
        for total_exc in range(
            1, self.num_exc + 1
        ):  # No excitation number conservation as in [1]
            previous_vector = np.zeros(self.number_degrees_freedom)
            previous_vector[0] = total_exc
            vector_list.append(previous_vector)
            while (
                previous_vector[-1] != total_exc
            ):  # step through until the last entry is total_exc
                next_vector = self.generate_next_vector(previous_vector, total_exc)
                vector_list.append(next_vector)
                previous_vector = next_vector
        return np.array(vector_list)

    def ptrace(self, density_matrix, keep_idxs):
        basis_vectors = self.gen_basis_vectors()
        num_keep_idxs = len(keep_idxs)
        remove_idxs = np.array(
            [idx for idx in range(self.number_degrees_freedom) if idx not in keep_idxs]
        )
        red_dim = int(comb(self.num_exc + num_keep_idxs, num_keep_idxs))
        new_dm = np.zeros((red_dim, red_dim), dtype=complex)
        new_basis_vecs = self.gen_basis_vectors(num_keep_idxs, self.num_exc)
        new_tags, new_index_array = self._gen_tags(new_basis_vecs)
        for row_idx in range(self.hilbert_dim()):
            for col_idx in range(row_idx, self.hilbert_dim()):
                ket = basis_vectors[row_idx, :]
                bra = basis_vectors[col_idx, :]
                # only if the two masks are the same does this
                # information survive the partial trace
                if np.allclose(ket.take(remove_idxs), bra.take(remove_idxs)):
                    new_ket_tag = self.hash(ket.take(keep_idxs))
                    new_bra_tag = self.hash(bra.take(keep_idxs))
                    ket_index = np.searchsorted(new_tags, new_ket_tag)
                    bra_index = np.searchsorted(new_tags, new_bra_tag)
                    new_dm[
                        new_index_array[ket_index], new_index_array[bra_index]
                    ] += density_matrix[row_idx, col_idx]
                    if row_idx != col_idx:
                        new_dm[
                            new_index_array[bra_index], new_index_array[ket_index]
                        ] += density_matrix[col_idx, row_idx]
        return Qobj(new_dm)

    @staticmethod
    def generate_next_vector(prev_vec: ndarray, radius: int) -> ndarray:
        """Algorithm for generating all vectors with positive entries of a given Manhattan length, specified in
        [1] J. M. Zhang and R. X. Dong, European Journal of Physics 31, 591 (2010)"""
        k = 0
        for num in range(len(prev_vec) - 2, -1, -1):
            if prev_vec[num] != 0:
                k = num
                break
        next_vec = np.zeros_like(prev_vec)
        next_vec[0:k] = prev_vec[0:k]
        next_vec[k] = prev_vec[k] - 1
        next_vec[k + 1] = radius - np.sum([next_vec[i] for i in range(k + 1)])
        return next_vec

    def a_operator(self, i: int) -> Qobj:
        """Construct the lowering operator for mode `i`.

        Parameters
        ----------
        i: int
            integer specifying the mode whose annihilation operator we would like to construct

        Returns
        -------
        Qobj
        """
        basis_vectors = self.gen_basis_vectors()
        tags, index_array = self._gen_tags(basis_vectors)
        dim = self.hilbert_dim()
        a = np.zeros((dim, dim))
        for w, vec in enumerate(basis_vectors):
            if vec[i] >= 1:
                temp_coefficient = np.sqrt(vec[i])
                basis_index = self._find_lowered_vector(vec, i, tags, index_array)
                if (
                    basis_index is not None
                ):  # Should not be the case here, only an issue for charge basis
                    a[basis_index, w] = temp_coefficient
        return Qobj(a)

    def trace_out_dict(self, state_dict, keep_idxs):
        return {
            label: self.ptrace(final_state, keep_idxs)
            for label, final_state in state_dict.items()
        }

    def _find_lowered_vector(
        self,
        vector: ndarray,
        i: int,
        tags: ndarray,
        index_array: ndarray,
        raised_or_lowered="lowered",
    ) -> Optional[int]:
        if raised_or_lowered == "lowered":
            pm_1 = -1
        elif raised_or_lowered == "raised":
            pm_1 = +1
        else:
            raise ValueError("only raised or lowered recognized")
        temp_vector = np.copy(vector)
        temp_vector[i] = vector[i] + pm_1
        temp_vector_tag = self.hash(temp_vector)
        index = np.searchsorted(tags, temp_vector_tag)
        if not np.allclose(tags[index], temp_vector_tag):
            return None
        basis_index = index_array[index]
        return basis_index

    def hilbert_dim(self) -> int:
        """Using the global excitation scheme the total number of states
        is given by the hockey-stick identity"""
        return int(
            comb(
                self.num_exc + self.number_degrees_freedom, self.number_degrees_freedom
            )
        )

    def hash(self, vector: ndarray) -> ndarray:
        """Generate the (unique) identifier for a given vector `vector`"""
        dim = len(vector)
        return np.sum(self.sqrt_prime_list[0:dim] * vector)

    def _gen_tags(self, basis_vectors: ndarray) -> Tuple[ndarray, ndarray]:
        """Generate the identifiers for all basis vectors `basis_vectors`"""
        dim = basis_vectors.shape[0]
        tag_list = np.array([self.hash(basis_vectors[i, :]) for i in range(dim)])
        index_array = np.argsort(tag_list)
        tag_list = tag_list[index_array]
        return tag_list, index_array

class Gates:
    def __init__(self, dim_cav, dim_q, ga, gb, gd1, gd2, gad, gdb, phi_a, phi_b, phi_d1, phi_d2, phi_ad, phi_db, delta_a, delta_b, delta_d, delta_ad, delta_db,
                 chi_a, chi_b, chi_ad, chi_db, T1_cav, Tphi_cav, T1_q, Tphi_q, kappa_bus, kappa_busphi, chi_a1a2, chi_a2d1, chi_d2b1, chi_b1b2) -> None:
        self.dim_cav = dim_cav
        self.dim_q = dim_q
        # Coupling strength
        self.ga = ga
        self.gb = gb
        self.gd1 = gd1
        self.gd2 = gd2
        self.gad = gad
        self.gdb = gdb
        # Phase of beamsplitter
        self.phi_a = phi_a
        self.phi_b = phi_b
        self.phi_d1 = phi_d1
        self.phi_d2 = phi_d2
        self.phi_ad = phi_ad
        self.phi_db = phi_db
        # Effective detuning between two modes
        self.delta_a = delta_a
        self.delta_b = delta_b
        self.delta_d = delta_d
        self.delta_ad = delta_ad
        self.delta_db = delta_db
        # dispersive coupling between transmon ancilla and cavity
        self.chi_a = chi_a
        self.chi_b = chi_b
        self.chi_ad = chi_ad
        self.chi_db = chi_db
        # T1, Tphi
        self.T1_cav = T1_cav
        self.Tphi_cav = Tphi_cav
        self.T1_q = T1_q
        self.Tphi_q = Tphi_q
        self.kappa_bus = kappa_bus
        self.kappa_busphi = kappa_busphi
        self.chi_a1a2 = chi_a1a2
        self.chi_a2d1 = chi_a2d1
        self.chi_d2b1 = chi_d2b1
        self.chi_b1b2 = chi_b1b2

    def ops(self, num_ex = 4, dof = 7):
        """number_ex: number of excitations in system; 
        dof: total number of elements in the entire system
        This function returns all operators needed in simulations, including cavity, bus, ancilla opeerators

        a_operators order: a1,a2,d1,d2,b1,b2,q1,q2
        q_operators order: q1,q2,sz1,sz2
        """
        hashing_instance = Hashing(num_exc = num_ex, number_degrees_freedom = dof)
        h_him = hashing_instance.hilbert_dim()
        a_operators = [hashing_instance.a_operator(i) for i in range(7)]
        # Tensor with two ancilla qubits & convert to sparse matrix
        for i in range(len(a_operators)):
            a_operators[i] = Qobj(tensor(a_operators[i], qeye(self.dim_q), qeye(self.dim_q)).to('csr'))
        # a_ops_labels = ["a1", "a2", "d1", "d2", "b1", "b2", "bus"]
        # labeled_a_ops = {label: op for label, op in zip(a_ops_labels, a_operators)}
        q1 = Qobj(tensor(qeye(h_him), destroy(self.dim_q), qeye(self.dim_q)).to('csr')) # connected to a2
        q2 = Qobj(tensor(qeye(h_him), qeye(self.dim_q), destroy(self.dim_q)).to('csr')) # connected to b1
        sz1 = Qobj(tensor(qeye(h_him), sigmaz(), qeye(self.dim_q)).to('csr'))
        sz2 = Qobj(tensor(qeye(h_him), qeye(self.dim_q), sigmaz()).to('csr'))
        q_operators = [q1, q2, sz1, sz2]
        # print(a_ops_labels)
        return a_operators, q_operators
    
    
    def c_ops(self):
        """This function returns all required collapse operators
            order: a1,a2,d1,d2,b1,b2,bus,q1,q2
        """
        a_ops, q_ops = self.ops()
        c1_ops = []
        cphi_ops = []
        for i in range(len(a_ops)-1):
            c1_ops.append(np.sqrt(1/self.T1_cav) * a_ops[i])
            cphi_ops.append(np.sqrt(1/self.Tphi_cav) * a_ops[i].dag() * a_ops[i])
        bus = a_ops[-1]
        c_bus1 = np.sqrt(self.kappa_bus) * bus # bus
        c_bus_phi = np.sqrt(self.kappa_busphi) * bus.dag() * bus
        c1_ops.append(c_bus1)
        cphi_ops.append(c_bus_phi)
        for i in range(2):
            c1_ops.append(np.sqrt(1/self.T1_q) * q_ops[i])
            cphi_ops.append(np.sqrt(1/self.Tphi_q) * q_ops[i].dag() * q_ops[i])
        return c1_ops, cphi_ops
    
    def gen_Z_op(self, mode, theta, num_ex = 4, dof = 7):
        """
        Generates the Z operator at mode i
        mode: index of mode i (6 cavities + 1 bus)
        output: Z operator of mode i
        """
        hashing_instance = Hashing(num_exc = num_ex, number_degrees_freedom = dof)
        basis_vecs = hashing_instance.gen_basis_vectors()
        z_op = qt.qzero(len(basis_vecs))
        for i in range(len(basis_vecs)):
            basis = np.zeros(len(basis_vecs))
            basis[i] = 1
            basis = Qobj(basis)
            if basis_vecs[i][mode] == 1:
                z_op += np.exp(1j * theta) * basis * basis.dag()
            else:
                z_op += basis * basis.dag()
        return tensor(z_op, qeye(self.dim_q), qeye(self.dim_q)).to('csr')
    
    def Hamiltonians(self, gate_name):
        """
        Local beamsplitter Hamiltonian
        dr_name: Name of dual rail qubit (Alice, Bob, or David)
        """
        a_ops, q_ops = self.ops()
        if gate_name == 'Alice':
            a1, a2 = a_ops[0], a_ops[1]
            H = self.ga/2 * (a1.dag()*a2*np.exp(1j*self.phi_a) + a1*a2.dag()*np.exp(-1j*self.phi_a)) + self.delta_a*a2.dag()*a2 + self.chi_a/2 *a2.dag()*a2*q_ops[2]
            return H
        elif gate_name == 'Bob':
            b1, b2 = a_ops[4], a_ops[5]
            H = self.gb/2 * (b1.dag()*b2*np.exp(1j*self.phi_b) + b1*b2.dag()*np.exp(-1j*self.phi_b)) + self.delta_b*b1.dag()*b1 + self.chi_b/2 *b1.dag()*b1*q_ops[3]
            return H
        elif gate_name == 'David':
            d1, d2, b = a_ops[2], a_ops[3], a_ops[6]
            H = self.gd1 * d1 * b.dag() * np.exp(1j*self.phi_d1) + self.gd1 * b * d1.dag() * np.exp(-1j*self.phi_d1) + self.gd2 * d2 * b.dag() * np.exp(1j*self.phi_d2) + self.gd2 * b * d2.dag() * np.exp(-1j*self.phi_d2) - self.delta_d * b.dag() * b
            return H
        # elif gate_name == 'ZZ_Teoh_AD':
        else:
            raise Exception('Invalid dr_name. dr_name should be Alice, Bob, or David')
        
    def logical_to_physical_dr(self, logical_state, num_ex = 4, dof = 7):
        """
        Input: Logical state list. e.g. |010>_L
        Output: State vector in the new Hilbert space
        """
        hashing_instance = Hashing(num_exc = num_ex, number_degrees_freedom = dof)
        h_him = hashing_instance.hilbert_dim()
        basis_vecs = hashing_instance.gen_basis_vectors()
        physical_state = [0]*7 # Initialize
        for i in range(len(logical_state)):
            if logical_state[i]== '0':
                physical_state[2*i], physical_state[2*i+1] = 0.,1.
            elif logical_state[i]== '1':
                physical_state[2*i], physical_state[2*i+1] = 1.,0.
            else:
                raise NameError('Each logical dual rail state must be either 0 or 1')
        # print(physical_state)
        idx_ = np.where((basis_vecs == physical_state).all(axis = 1))[0][0]
        state = np.zeros(h_him)
        state[idx_] = 1
        return Qobj(state).unit().to('csr')
    
    def gate_time(self, gate_name):
        """
        Input: Gate name (options: H_A, H_B, H_D_VR, H_D_onres, ZZ_SWS, ZZ_Teoh)
        Output: Gate time (in us)
        """
        if gate_name == 'H_A':
            return np.pi/(2*self.ga)
        elif gate_name == 'H_B':
            return np.pi/(2*self.gb)
        elif gate_name == 'H_D_VR':
            return np.pi*self.delta_d/(4*self.gd1**2)
        elif gate_name == 'H_D_onres':
            return float(np.sqrt(3/8)*np.pi/self.gd1)
        else:
            raise NameError('gate_name should be one of the following: H_A, H_B, H_D_VR, H_D_onres, ZZ_SWS, ZZ_Teoh')
        
    def truncated_matrix(self, input_ds, basis, num_ex = 4, dof = 7):
        """
        basis: array that indicates truncated basis, e.g.[[0,1,0,1,0,0,0], [0,1,1,0,0,0,0], [1,0,0,1,0,0,0],[1,0,1,0,0,0,0]]
        output truncated matrix with respect to desired truncated basis
        """
        hashing_instance = Hashing(num_exc = num_ex, number_degrees_freedom = dof)
        h_him = hashing_instance.hilbert_dim()
        basis_vecs = hashing_instance.gen_basis_vectors()
        idx_arr = []
        for i in range(len(basis)):
            idx_base = np.where((basis_vecs == basis[i, :]).all(axis = 1))[0][0]
            idx_arr.append(idx_base)
        input_ds_arr = input_ds.full()
        truncated_mat = np.zeros((len(basis), len(basis)), dtype = complex)
        for i in range(len(idx_arr)):
            for j in range(len(idx_arr)):
                truncated_mat[i][j] = input_ds_arr[idx_arr[i]][idx_arr[j]]
        return Qobj(truncated_mat)
    
    def ZZ_matrix(self):
        """
        output
        """
        
    def CZ(self, input_state):
        """
        Output: final_state after performing ZZ(pi/2) gate
        protocol: either 'Teoh' or 'SWS'
        target: Dual rails that ZZ gate is performing on. Should be either AD (Alice & David) or DB (David Bob)
        """
        hashing_instance = Hashing(num_exc = 4, number_degrees_freedom = 7)
        h_him = hashing_instance.hilbert_dim()
        aops, qops = self.ops()
        # if target == 'AD':
        a1, b1, sz1 = aops[1], aops[2], qops[2] # a2, d1
        g1, phi1, delta1, chi1 = self.gad, self.phi_ad, self.delta_ad, self.chi_db
        # elif target == 'DB':
        a2, b2, sz2 = aops[3], aops[4], qops[3] # d2, b1
        g2, phi2, delta2, chi2 = self.gdb, self.phi_db, self.delta_db, self.chi_db
        # else:
            # raise NameError('input target should be either "AD" or "DB". ')
        # if protocol == 'Teoh':
        H = g1/2 * (a1.dag() * b1 * np.exp(1j * phi1) +  b1.dag() * a1 * np.exp(-1j * phi1)) + delta1 * a1.dag() * a1 + chi1/2 * a1.dag() * a1 * sz1 \
            + g2/2 * (a2.dag() * b2 * np.exp(1j * phi2) +  b2.dag() * a2 * np.exp(-1j * phi2)) + delta2 * a2.dag() * a2 + chi2/2 * a2.dag() * a2 * sz2
        # else:
            # raise NameError('protocol should be either Teoh or SWS.')
        if input_state.isket == False:
            # Step 1: ZZ(pi/2) gate on a2 and d1
            # Y(pi/2) qubit drive
            opts = Options(atol = 1e-10, rtol = 1e-10, store_final_state = True)
            # qubit_state = ry(np.pi/2) * fock(self.dim_q, 0)
            # qubit_state = (fock(2,0) + fock(2,1)).unit()
            # init_state = tensor(ptrace(input_state, 0), qubit_state * qubit_state.dag(), fock(self.dim_q, 0) * fock(self.dim_q, 0).dag()).to('csr')
            init_state = tensor(qeye(h_him), ry(np.pi/2), ry(np.pi/2)) * input_state * tensor(qeye(h_him), ry(np.pi/2), ry(np.pi/2)).dag()
            # First joint parity
            tlist = np.linspace(0, 2*np.pi/chi1, 100)
            final_state = mesolve(H, init_state, tlist).final_state
            final_state = final_state.to('csr')
            # X(pi/2) 
            X_pi2 = tensor(qeye(h_him), rx(np.pi/2), rx(np.pi/2)).to('csr')
            final_state = X_pi2 * final_state * X_pi2.dag()
            # 2nd joint parity
            final_state = mesolve(H, final_state, tlist).final_state
            final_state = final_state.to('csr')
            # Y(-pi/2) rotation
            Y_npi2 = tensor(qeye(h_him), ry(-np.pi/2), ry(-np.pi/2)).to('csr')
            final_state = Y_npi2 * final_state * Y_npi2.dag()
            
            # Step 2: Add phase gate -pi/2 on a2 and -pi/2 on d1. Now we have a physical CZ gate on a2 and d1
            final_state = self.gen_Z_op(1, -np.pi/2) * final_state * self.gen_Z_op(1, -np.pi/2).dag()
            final_state = self.gen_Z_op(2, -np.pi/2) * final_state * self.gen_Z_op(2, -np.pi/2).dag()

            final_state = self.gen_Z_op(3, -np.pi/2) * final_state * self.gen_Z_op(3, -np.pi/2).dag()
            final_state = self.gen_Z_op(4, -np.pi/2) * final_state * self.gen_Z_op(4, -np.pi/2).dag()

            # Step 3: Add single qubit phase gate on a2 and d1 to make logical CZ gate on Alice and David
            final_state = self.gen_Z_op(1, np.pi) * final_state * self.gen_Z_op(1, np.pi).dag()
            final_state = self.gen_Z_op(2, 0) * final_state * self.gen_Z_op(2, 0).dag()

            final_state = self.gen_Z_op(3, np.pi) * final_state * self.gen_Z_op(3, np.pi).dag()
            final_state = self.gen_Z_op(4, 0) * final_state * self.gen_Z_op(4, 0).dag()

            return final_state * np.exp(1j * np.pi/2)
        else:
            # qubit_state = (fock(2,0) + fock(2,1)).unit()
            # init_state = tensor(input_state, qubit_state, qubit_state).to('csr')
            init_state = tensor(qeye(h_him), ry(np.pi/2), ry(np.pi/2)) * input_state
            tlist = np.linspace(0, 2*np.pi/chi1, 100)
            final_state = mesolve(H, init_state, tlist).final_state
            final_state = final_state.to('csr')

            X_pi2 = tensor(qeye(h_him), rx(np.pi/2), rx(np.pi/2)).to('csr')
            final_state = X_pi2 * final_state

            final_state = mesolve(H, final_state, tlist).final_state
            final_state = final_state.to('csr')

            Y_npi2 = tensor(qeye(h_him), ry(-np.pi/2), ry(-np.pi/2)).to('csr')
            final_state = Y_npi2 * final_state
            final_state = self.gen_Z_op(1, -np.pi/2) * final_state
            final_state = self.gen_Z_op(2, -np.pi/2) * final_state

            final_state = self.gen_Z_op(3, -np.pi/2) * final_state
            final_state = self.gen_Z_op(4, -np.pi/2) * final_state

            final_state = self.gen_Z_op(1, np.pi) * final_state
            final_state = self.gen_Z_op(2, 0) * final_state

            final_state = self.gen_Z_op(3, np.pi) * final_state
            final_state = self.gen_Z_op(4, 0) * final_state

            return final_state * np.exp(1j * np.pi/2)


        
    def simulate(self, num_ex = 4, dof = 7, david_had = 'H_D_VR', decoherence = True):
        """
        Simulate the protocol for the generation of Bell Pair
        """
        # Step 1: Hadamard gate on Alice, David, and Bob
        hashing_instance = Hashing(num_exc = num_ex, number_degrees_freedom = dof)
        basis_vecs = hashing_instance.gen_basis_vectors()
        h_him = hashing_instance.hilbert_dim()

        # Initial State
        init_state = self.logical_to_physical_dr(['0', '0', '0'])
        init_state = tensor(Qobj(init_state), fock(self.dim_q, 0), fock(self.dim_q, 0)).unit().to('csr')

        # Stage 0: Hadamard gate on Alice, Bob, and David from t = 0 to t = np.pi/(2g)
        H0 = self.Hamiltonians('Alice') + self.Hamiltonians('Bob') + self.Hamiltonians('David')
        tlist0 = np.linspace(0, self.gate_time('H_A'), 1000)
        opts = Options(atol = 1e-10, rtol = 1e-10, store_final_state = True)
        final_state = mesolve(H0, init_state, tlist0).final_state

        # Stage 1: Hadamard gate on A & B completed, they wait. David continues on Hadamard gate
        H1 = self.Hamiltonians('David')
        tfinal_1 = self.gate_time(david_had) - self.gate_time('H_A')
        tlist1 = np.linspace(0, tfinal_1, 1000)
        final_state = mesolve(H1, final_state, tlist1).final_state

        # Step 2: CZ gate on a2, d1 and on d2, b1
        final_state = self.CZ(final_state.to('csr'))

        # Step 3: Hadamard Gate on David
        H2 = self.Hamiltonians('David')
        tlist2 = np.linspace(0, self.gate_time('H_D_VR'), 2000)
        final_state = mesolve(H2, final_state, tlist2).final_state

        return final_state

