from abc import ABC, abstractmethod
import numpy as np
from qcelemental.models.molecule import Molecule

class TaskDriver(ABC):
    def __init__(self):
        pass

    def setup_env(self):
        pass

    """
        Abstract methods are given a qcelement molecule as input.
    """

    @abstractmethod
    def energy(self, mol:Molecule) -> float:
        pass
    
    @abstractmethod
    def gradient(self, mol:Molecule) -> np.ndarray:
        pass
    
    @abstractmethod
    def hessian(self, mol:Molecule) -> np.ndarray:
        pass
    


