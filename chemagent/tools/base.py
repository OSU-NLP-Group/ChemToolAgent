from abc import ABC, abstractmethod
import logging


logger = logging.getLogger(__name__)


class BaseTool(ABC):
    name: str
    func_name: str
    description: str
    func_doc: tuple
    func_description: str
    examples: list

    def __init__(self, init=True, interface='text') -> None:
        assert interface in ('text', 'code'), "Interface '%s' is not supported. Please use 'text' or 'code'." % interface
        self.interface = interface
        super().__init__()
        if init:
            self._init_modules()

    def _init_modules(self):
        pass

    def __call__(self, *args, **kwargs):
        logger.debug("===== Starting tool {} =====".format(self.__class__.name))
        if self.interface == 'text':
            r = self.run_text(args[0])
        elif self.interface == 'code':
            r = self.run_code(*args, **kwargs)
        else:
            raise NotImplementedError("Interface '%s' is not supported. Please use 'text' or 'code'." % self.interface)
        logger.debug("----- Ending tool {} -----".format(self.__class__.name))
        return r

    def run_text(self, query, *args, **kwargs):
        return self._run_text(query, *args, **kwargs)
    
    def run_code(self, *args, **kwargs):
        return self._run_code(*args, **kwargs)

    def _run_text(self, query, *args, **kwargs):
        return str(self._run_base(query, *args, **kwargs))
    
    def _run_code(self, *args, **kwargs):
        return self._run_base(*args, **kwargs)
    
    @abstractmethod
    def _run_base(self, *args, **kwargs):
        raise NotImplementedError

    def run(self, query, *args, **kwargs):
        raise DeprecationWarning("The run function is deprecated. Please modify the implementation.")
