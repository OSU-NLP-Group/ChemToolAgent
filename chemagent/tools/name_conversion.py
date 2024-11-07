import logging

import selfies as sf
import pubchempy as pcp
from rdkit import Chem
import rdkit.Chem.rdMolDescriptors as molD

from chemagent.utils.error import *
from chemagent.tools import BaseTool, ChemSpace
from chemagent.utils import (
    # canonicalize_molecule_smiles,
    is_smiles, 
    is_multiple_smiles,
    # largest_mol,
    pubchem_iupac2cid,
    pubchem_name2cid,
)


logger = logging.getLogger(__name__)


def pubchem_iupac2smiles(
    query: str,
) -> str:
    cid = pubchem_iupac2cid(query)
    if not isinstance(cid, tuple):
        cid = (cid,)
    smiles = []
    for single_cid in cid:
        c = pcp.Compound.from_cid(single_cid)
        r = c.isomeric_smiles
        smiles.append(r)
    r = '.'.join(smiles)

    return r


def pubchem_name2smiles(
    query: str,
) -> str:
    cid = pubchem_name2cid(query)
    c = pcp.Compound.from_cid(cid)
    r = c.isomeric_smiles

    return r


def pubchem_smiles2iupac(smi):
    """This function queries the given molecule smiles and returns iupac"""

    c = pcp.get_compounds(smi, 'smiles')
    
    if len(c) == 0:
        parts = smi.split('.')
        if len(parts) > 1:
            parts_iupac = []
            parts_cannot_find = []
            for part in parts:
                try:
                    iupac = pubchem_smiles2iupac(part)
                except ChemAgentSearchError:
                    parts_cannot_find.append(part)
                else:
                    parts_iupac.append(iupac)
            if len(parts_cannot_find) > 0:
                raise ChemAgentSearchError("Cannot find a matched molecule/compound for the following parts of the input SMILES: %s" % ', '.join(parts_cannot_find))
            else:
                r = ';'.join(parts_iupac)
        else:
            raise ChemAgentSearchError("Cannot find a matched molecule/compound. Please check the input SMILES.")
    elif len(c) >= 1:
        if len(c) > 1:
            logger.info("There are more than one molecules/compounds that match the input SMILES. Using the first matched one.")
        c = c[0]
    
        r = c.iupac_name
    
    if r is None or r == 'None':
        raise ValueError("The IUPAC name is None. The cid is %s" % c.cid)

    return r


def addHs(mol):
    mol = Chem.rdmolops.AddHs(mol, explicitOnly=True)
    return mol


def smiles2formula(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    mol = addHs(mol)
    formula = molD.CalcMolFormula(mol)
    return formula


class IUPAC2SMILES(BaseTool):
    name = "IUPAC2SMILES"
    func_name = 'convert_iupac_to_smiles'
    description = "Input IUPAC name of molecule/compound (one at a time), returns SMILES. To get SMILES from IUPAC name, you must use this tool."
    func_doc = ("iupac: str", "str")
    func_description = description
    examples = [
        {'input': 'ethanol', 'output': 'CCO'},
    ]

    def __init__(self, chemspace_api_key: str = None, init=True, interface='text'):
        self.translate_reverse = None
        super().__init__(init, interface=interface)
        self.chemspace_api_key = chemspace_api_key
    
    def _init_modules(self):
        from STOUT import translate_reverse
        self.translate_reverse = translate_reverse

    def _run_base(self, query: str, *args, **kwargs) -> str:
        try:
            try:
                smi = pubchem_iupac2smiles(query)
                logger.debug("Looking up PubChem succeeded.")
            except ChemAgentSearchError as e:
                logger.debug("Looking up PubChem failed.")
                if self.chemspace_api_key:
                    chemspace = ChemSpace(self.chemspace_api_key)
                    smi = chemspace.convert_mol_rep(query, "smiles")
                    try:
                        tmp = smi.split(":")[1]
                    except IndexError:
                        logger.debug("Looking up ChemSpace failed, due to IndexError.")
                        logger.debug("The result obtained from ChemSpace is: %s" % smi)
                        raise e
                    else:
                        smi = tmp
                    logger.debug("Looking up ChemSpace succeeded.")
                else:
                    logger.debug("Looking up ChemSpace failed, because ChemSpace API is not set.")
                    raise
        except ChemAgentSearchError:
            try:
                if self.translate_reverse is None:
                    self._init_modules()
                smi = self.translate_reverse(query)
                smi = smi + '\nNote: The result is predicted by neural networks and is possibly incorrect or inaccurate.'
            except:
                logger.debug("Using STOUT failed.")
                raise ChemAgentToolProcessError("Error: Cannot get the SMILES of the input IUPAC name.")
                # raise
            else:
                logger.debug("Using STOUT succeeded.")

        return smi
    

class SMILES2IUPAC(BaseTool):
    name = "SMILES2IUPAC"
    func_name = 'convert_smiles_to_iupac'
    description = "Input SMILES of a molecule/compound (one at a time), returns IUPAC name. To get IUPAC name from SMILES, you must use this tool."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'ethanol'},
    ]

    def __init__(self, init=True, interface='text') -> None:
        self.translate_forward = None
        super().__init__(init, interface=interface)

    def _init_modules(self):
        from STOUT import translate_forward
        self.translate_forward = translate_forward

    def _run_base(self, query: str, *args, **kwargs) -> str:
        """Use the tool."""
        if not is_smiles(query):
            if '.' in query:
                parts = query.split('.')
                parts_invalid = []
                for part in parts:
                    if not is_smiles(part):
                        parts_invalid.append(part)
                raise ChemAgentInputError("The input is not valid SMILES. Please double check. The following parts of the input are invalid: %s" % ', '.join(parts_invalid))
            else:
                raise ChemAgentInputError("The input is not valid SMILES. Please double check.")
        
        try:
            name = pubchem_smiles2iupac(query)
            logger.debug("Looking up PubChem succeeded.")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.debug("Looking up PubChem failed.")
            try:
                if self.translate_forward is None:
                    self._init_modules()
                name = self.translate_forward(query)
                name = name + '\nNote: The result is predicted by neural networks and is possibly incorrect or inaccurate.'
                logger.debug("Using STOUT succeeded.")
            except (KeyboardInterrupt, ImportError):
                raise
            except Exception as e2:
                logger.debug("Using STOUT failed.")
                raise ChemAgentToolProcessError("Cannot convert the input to IUPAC name. Looking up on PubChem and using STOUT both failed. Looking up failed because: %s Using STOUT failed because: %s Please double check the input or try another method." % (str(e), str(e2)))
        
        return name


class SMILES2Formula(BaseTool):
    name = 'SMILES2Formula'
    func_name = 'convert_smiles_to_molecular_formula'
    description = "Input SMILES of a molecule/compound (one at a time), returns molecular formula. To get molecular formula from SMILES, you must use this tool."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'C2H6O'},
    ]

    def _run_base(self, query: str, *args, **kwargs) -> str:
        """Use the tool."""
    
        try:
            if not is_smiles(query):
                raise ChemAgentInputError("The input is not valid SMILES. Please double check. If there are multiple parts in the input SMILES, please use dot as the separator.")
            formula = smiles2formula(query)
            return formula
        except Exception as e:
            raise ChemAgentToolProcessError(str(e))


class Name2SMILES(BaseTool):
    name = "Name2SMILES"
    func_name = 'convert_chemical_name_to_smiles'
    description = "Input common name of molecule/compound (one at a time), returns SMILES."
    func_doc = ("name: str", "str")
    func_description = description
    examples = [
        {'input': 'aspirin', 'output': 'CC(=O)OC1=CC=CC=C1C(=O)O'},
    ]

    def __init__(self, init=True, interface='text'):
        self.translate_reverse = None
        super().__init__(init, interface=interface)

    def _run_base(self, query: str, *args, **kwargs) -> str:
        smi = pubchem_name2smiles(query)
        return smi


class SELFIES2SMILES(BaseTool):
    name = "SELFIES2SMILES"
    func_name = 'convert_selfies_to_smiles'
    description = (
        "Input SELFIES representation, returns SMILES representation."
    )
    func_doc = ("selfies: str", "str")
    func_description = description
    examples = [
        {'input': '[C][C][O]', 'output': 'CCO'},
    ]

    def _run_base(self, selfies: str, *args, **kwargs) -> str:
        try:
            smiles = sf.decoder(selfies)
        except KeyboardInterrupt:
            raise
        except:
            raise ChemAgentToolProcessError("Cannot convert the SELFIES into SMILES, possibly because it is not a valid SELFIES string.")
        return smiles


class SMILES2SELFIES(BaseTool):
    name = "SMILES2SELFIES"
    func_name = 'convert_smiles_to_selfies'
    description = (
        "Input SMILES representation, returns SELFIES representation."
    )
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': '[C][C][O]'},
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        if not is_smiles(smiles):
            raise ChemAgentInputError("The input is not valid SMILES. Please double check.")
        try:
            selfies = sf.encoder(smiles)
        except KeyboardInterrupt:
            raise
        except:
            raise ChemAgentToolProcessError("Cannot convert the SMILES into SELFIES, possibly because it is not a valid SMILES string.")
        return selfies

