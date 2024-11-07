from transformers import T5Tokenizer, T5ForConditionalGeneration

from .base import BaseTool
from chemagent.utils.smiles import is_smiles
from chemagent.utils.error import *


class MoleculeCaptioner(BaseTool):
    name = "MoleculeCaptioner"
    func_name = "generate_molecule_caption"
    description = "Input the SMILES of a molecule/compound, returns the textual description of the molecule/compound. This tool uses neural networks to generate descriptions, which may not be accurate or correct. You should try the PubchemSearch or WebSearch tool first, which provide accurate and authoritative information, and only use this one when other tools cannot provides useful information."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'The molecule is an ether in which the oxygen atom is linked to two ethyl groups. It has a role as an inhalation anaesthetic, a non-polar solvent and a refrigerant. It is a volatile organic compound and an ether.'},
    ]

    def __init__(self, init=True, interface='text') -> None:
        self.tokenizer, self.model = None, None
        super().__init__(init, interface=interface)

    def _init_modules(self):
        self.tokenizer, self.model = self.__load_molt5()
        
    def __load_molt5(self):
        tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-smiles2caption", model_max_length=1024)
        model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-smiles2caption')
        return tokenizer, model
    
    def _run_molt5(self, smiles):
        if self.tokenizer is None or self.model is None:
            self._init_modules()
        input_ids = self.tokenizer(smiles, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, num_beams=5, max_length=1024)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    
    def _run_base(self, smiles, *args, **kwargs):
        if not is_smiles(smiles):
            raise ChemAgentInputError("The input is not a valid SMILES. Please double check.")
        
        return self._run_molt5(smiles)


class MoleculeGenerator(BaseTool):
    name = "MoleculeGenerator"
    func_name = "generate_molecule_from_description"
    description = "Input a description of a molecule/compound, returns the SMILES representation of the molecule/compound. This tool uses neural networks to generate molecules, which may not be accurate or correct."
    func_doc = ("description: str", "str")
    func_description = description
    examples = [
        {'input': 'The molecule is an ether in which the oxygen atom is linked to two ethyl groups. It has a role as an inhalation anaesthetic, a non-polar solvent and a refrigerant. It is a volatile organic compound and an ether.', 'output': 'CCO'},
    ]

    def __init__(self, init=True, interface='text') -> None:
        self.tokenizer, self.model = None, None
        super().__init__(init, interface=interface)

    def _init_modules(self):
        self.tokenizer, self.model = self.__load_molt5()
        
    def __load_molt5(self):
        tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
        model = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles')
        return tokenizer, model
    
    def _run_molt5(self, text):
        if self.tokenizer is None or self.model is None:
            self._init_modules()
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        outputs = self.model.generate(input_ids, num_beams=5, max_length=512)
        smiles = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return smiles
    
    def _run_base(self, description, *args, **kwargs):
        return self._run_molt5(description)
