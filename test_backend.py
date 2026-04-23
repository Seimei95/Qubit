from qiskit_ibm_runtime.fake_provider import FakeManilaV2
backend = FakeManilaV2()
try:
    props = backend.qubit_properties()
    print("Called with () ->", type(props))
except Exception as e:
    print("Error calling with ():", e)
    
try:
    props = backend.qubit_properties([0, 1])
    print("Called with ([0,1]) ->", type(props))
except Exception as e:
    print("Error calling with ([0,1]):", e)
