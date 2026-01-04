#Define the first class citizen: Chain complex with python class



class ChainComplex:    
    def __init__(self, degrees, differentials):
        pass


    def __repr__(self):
        pass



def compile_from_program(program):
    """
    Compile a chain complex from a given program representation.

    :param program: A representation of the chain complex in some programmatic form.
    :return: An instance of ChainComplex.
    """
    # Placeholder implementation; actual implementation would parse the program
    degrees = program.get('degrees', [])
    differentials = program.get('differentials', {})
    
    return ChainComplex(degrees, differentials)