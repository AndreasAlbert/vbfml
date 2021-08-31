import uproot

# fname = '/home/albert/repos/bucoffea/bucoffea/plot/treeplot/input/trees/2020-08-26_tree_2m2e_nodpfcalo_v3/tree_DYJetsToLL_M-50_HT-400to600-MLM_ext1_2017.root'

# f = uproot.open(fname)
# t=f['cr_2m_j']

from tests.test_datagen import create_test_tree

create_test_tree("test.root","tree", branches=['x','y'],n_events = 5)