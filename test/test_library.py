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
