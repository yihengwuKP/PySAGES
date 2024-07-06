#!/usr/bin/env python

import jax
import numpy as np
import openmm.app as app
from jax import grad

from pysages.colvars.orientation import ERMSD, ERMSDCG

pdb = app.PDBFile("../../../inputs/GAGA.box_0mM.pdb")
basetype2selectedAA = {
    "A": ["C8", "N6", "C2"],
    "U": ["C6", "O4", "O2"],
    "G": ["C8", "O6", "N2"],
    "C": ["C6", "N4", "O2"],
}

B123_indices = []

for i, res in enumerate(pdb.topology.residues()):
    if res.name in basetype2selectedAA.keys():
        B123_residue = dict.fromkeys(basetype2selectedAA[res.name])
        for atom in res.atoms():
            if atom.name in basetype2selectedAA[res.name]:
                B123_residue[atom.name] = atom.index
    B123_indices.append(B123_residue)

# notice that the order of the indices for eRMSD is tricky!
B123_indices_ordered = []
for i, res in enumerate(pdb.topology.residues()):
    if res.name in basetype2selectedAA.keys():
        B123 = B123_indices[i]
        B123_indices_ordered.extend((B123[Bname] for Bname in basetype2selectedAA[res.name]))

reference_CG = pdb.getPositions(asNumpy=True).astype("float")[np.asarray(B123_indices_ordered)]
sequence = [res.name for res in pdb.topology.residues() if res.name in "AUGC"]
nt2idx = {nt: idx for nt, idx in zip(["A", "U", "G", "C"], [0, 1, 2, 3])}
sequence = [nt2idx[s] for s in sequence]

ermsd = ERMSDCG(B123_indices_ordered, reference_CG, sequence, cutoff=3.2)
ermsd_grad = grad(ermsd.function)
print(ermsd_grad(reference_CG + np.random.random(reference_CG.shape)))


# def G(r, cutoff=2.4):
#    gamma = jax.numpy.pi / cutoff
#    end = jax.numpy.sin(gamma * r) * jax.numpy.heaviside(cutoff - r, np.zeros(r.shape))
#    return end[0]
#
#
# print(grad(G)(np.ones((10,))))
