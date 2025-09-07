from abc import ABC, abstractmethod

class Acquisition(ABC):
    
    @abstractmethod
    def readBlock(self, n):
        """ Read n samples and return as an array """
        pass
    
    @abstractmethod
    def readStream(self):
        """ Generate samples indefinitely """
        pass
    