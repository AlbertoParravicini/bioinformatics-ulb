# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:06:11 2017

@author: albyr
"""

import Bio.PDB
import matplotlib.pyplot as plt
import numpy as np

pdb_code = "1XI4"
pdb_filename = "C:\\Users\\albyr\\OneDrive\\Documenti\\ULB\\First Year\\Bioinformatics\\final-presentation\\python-example\\1XI4.pdb" #not the full cage!

structure = Bio.PDB.PDBParser().get_structure(pdb_code, pdb_filename)
model = structure[0]

def calc_residue_dist(residue_one, residue_two):
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sum(diff_vector * diff_vector)

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    answer = np.zeros((len(chain_one), len(chain_two)), np.float)
    for row, residue_one in enumerate(chain_one):
        for col, residue_two in enumerate(chain_two):
            answer[row, col] = calc_residue_dist(residue_one, residue_two)
    return answer

dist_matrix = calc_dist_matrix(model["M"], model["M"])
contact_map = dist_matrix < 12**2

plt.imshow(contact_map, cmap='hot', interpolation='nearest')
plt.show()


