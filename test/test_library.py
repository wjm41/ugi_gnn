from dock2hit.library_generation import enumerate_ugi, decompose_ugi


def test_generate_library():
    test_acids = []
    test_amines = []
    test_aldehydes = []
    test_isocyanides = []

    test_lib = enumerate_ugi.generate_ugi_library(
        test_acids, test_amines, test_aldehydes, test_isocyanides)
    assert len(test_lib) == len(test_acids) * len(test_amines) * \
        len(test_aldehydes) * len(test_isocyanides)


def test_decompose_ugi():
    best_ugi_mol = 'CC(C)Oc1ccc(cc1)N(C(C(=O)NCCc1ccns1)c1cccnc1)C(=O)c1cocn1'
    components = decompose_ugi.decompose_ugi_molecule_into_components(
        best_ugi_mol)
    assert len(components) == 4
    assert components[0] == 'O=C(O)c1cocn1'
    assert components[1] == 'CC(C)Oc1ccc(N)cc1'
    assert components[2] == 'O=Cc1cccnc1'
    assert components[3] == '[C-]#[N+]CCc1ccns1'
