from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from chemagent.utils.error import *
from chemagent.tools import BaseTool
from chemagent.utils import tanimoto, canonicalize_molecule_smiles, get_molecule_id, is_smiles


class MolSimilarity(BaseTool):
    name = "MolSimilarity"
    func_name = 'cal_molecule_similarity'
    description = (
        "Input two molecule SMILES (separated by ';'), returns Tanimoto similarity."
    )
    func_doc = ("smiles1: str, smiles2: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO;CCN', 'output': 'The Tanimoto similarity between CCO and CCN is 0.3333, indicating that the two molecules are not similar.'},
        {'input': 'CCO;CCO', 'output': 'Input Molecules Are Identical'},
    ]

    def _run_text(self, smiles_pair: str):
        smi_list = smiles_pair.split(";")
        if len(smi_list) != 2:
            raise ChemAgentInputError("Input error, please input exactly two SMILES strings separated by ';'")
        else:
            smiles1, smiles2 = smi_list
        return self._run_base(smiles1, smiles2)

    def _run_base(self, smiles1, smiles2, *args, **kwargs) -> str:
        similarity = tanimoto(smiles1, smiles2)

        if isinstance(similarity, str):
            return similarity

        sim_score = {
            0.9: "very similar",
            0.8: "similar",
            0.7: "somewhat similar",
            0.6: "not very similar",
            0: "not similar",
        }
        if similarity == 1:
            return "The input molecules are identical."
        else:
            val = sim_score[
                max(key for key in sim_score.keys() if key <= round(similarity, 1))
            ]
            message = f"The Tanimoto similarity between {smiles1} and {smiles2} is {round(similarity, 4)}, indicating that the two molecules are {val}."
        return message


class SMILES2Weight(BaseTool):
    name = "SMILES2Weight"
    func_name = 'cal_molecular_weight'
    description = "Calculate molecular weight. Input SMILES, returns molecular weight."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': '46.041864812'},
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ChemAgentInputError("Invalid SMILES string. Please make sure that you input a valid SMILES string, and only one at a time.")
        mol_weight = rdMolDescriptors.CalcExactMolWt(mol)
        return str(mol_weight)


class FuncGroups(BaseTool):
    name = "FunctionalGroups"
    func_name = 'get_functional_groups'
    description = "Get the functional groups in a molecule. Input SMILES, return list of functional groups in the molecule."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'This molecule contains alcohol groups, and side-chain hydroxyls.'},
    ]

    def __init__(
        self, init=True, interface='text'
    ):
        super().__init__(init, interface)

        # List obtained from https://github.com/rdkit/rdkit/blob/master/Data/FunctionalGroups.txt
        self.dict_fgs = {
            "furan": "o1cccc1",
            "aldehydes": " [CX3H1](=O)[#6]",
            "esters": " [#6][CX3](=O)[OX2H0][#6]",
            "ketones": " [#6][CX3](=O)[#6]",
            "amides": " C(=O)-N",
            "thiol groups": " [SH]",
            "alcohol groups": " [OH]",
            "methylamide": "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
            "carboxylic acids": "*-C(=O)[O;D1]",
            "carbonyl methylester": "*-C(=O)[O;D2]-[C;D1;H3]",
            "terminal aldehyde": "*-C(=O)-[C;D1]",
            "amide": "*-C(=O)-[N;D1]",
            "carbonyl methyl": "*-C(=O)-[C;D1;H3]",
            "isocyanate": "*-[N;D2]=[C;D2]=[O;D1]",
            "isothiocyanate": "*-[N;D2]=[C;D2]=[S;D1]",
            "nitro": "*-[N;D3](=[O;D1])[O;D1]",
            "nitroso": "*-[N;R0]=[O;D1]",
            "oximes": "*=[N;R0]-[O;D1]",
            "Imines": "*-[N;R0]=[C;D1;H2]",
            "terminal azo": "*-[N;D2]=[N;D2]-[C;D1;H3]",
            "hydrazines": "*-[N;D2]=[N;D1]",
            "diazo": "*-[N;D2]#[N;D1]",
            "cyano": "*-[C;D2]#[N;D1]",
            "primary sulfonamide": "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
            "methyl sulfonamide": "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
            "sulfonic acid": "*-[S;D4](=O)(=O)-[O;D1]",
            "methyl ester sulfonyl": "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
            "methyl sulfonyl": "*-[S;D4](=O)(=O)-[C;D1;H3]",
            "sulfonyl chloride": "*-[S;D4](=O)(=O)-[Cl]",
            "methyl sulfinyl": "*-[S;D3](=O)-[C;D1]",
            "methyl thio": "*-[S;D2]-[C;D1;H3]",
            "thiols": "*-[S;D1]",
            "thio carbonyls": "*=[S;D1]",
            "halogens": "*-[#9,#17,#35,#53]",
            "t-butyl": "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
            "tri fluoromethyl": "*-[C;D4](F)(F)F",
            "acetylenes": "*-[C;D2]#[C;D1;H]",
            "cyclopropyl": "*-[C;D3]1-[C;D2]-[C;D2]1",
            "ethoxy": "*-[O;D2]-[C;D2]-[C;D1;H3]",
            "methoxy": "*-[O;D2]-[C;D1;H3]",
            "side-chain hydroxyls": "*-[O;D1]",
            "ketones": "*=[O;D1]",
            "primary amines": "*-[N;D1]",
            "nitriles": "*#[N;D1]",
        }

    def _is_fg_in_mol(self, mol, fg):
        fgmol = Chem.MolFromSmarts(fg)
        mol = Chem.MolFromSmiles(mol.strip())
        return len(Chem.Mol.GetSubstructMatches(mol, fgmol, uniquify=True)) > 0

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        """
        Input a molecule SMILES or name.
        Returns a list of functional groups identified by their common name (in natural language).
        """
        try:
            fgs_in_molec = [
                name
                for name, fg in self.dict_fgs.items()
                if self._is_fg_in_mol(smiles, fg)
            ]
            if len(fgs_in_molec) > 1:
                return f"This molecule contains {', '.join(fgs_in_molec[:-1])}, and {fgs_in_molec[-1]}."
            else:
                return f"This molecule contains {fgs_in_molec[0]}."
        except:
            raise ChemAgentInputError("Wrong argument. Please input a valid molecular SMILES.")


class CompareSMILES(BaseTool):
    name = "CompareSMILES"
    func_name = 'check_molecule_identical'
    description = "Input two molecule SMILES (separated by ';'), returns if they are identical. To judge if two molecules are identical, you should always use this tool, instead of directly comparing the SMILES strings."
    func_doc = ("smiles1: str, smiles2: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO;CCN', 'output': 'different'},
        {'input': 'OCC;CCO', 'output': 'identical'},
    ]

    def _run_text(self, smiles_pair: str):
        smi_list = smiles_pair.split(';')
        if len(smi_list) != 2:
            raise ChemAgentInputError("Input error, please input two smiles strings separated by ';'")
        else:
            smiles1, smiles2 = smi_list
        return self._run_base(smiles1, smiles2)

    def _run_base(self, smiles1, smiles2, *args, **kwargs) -> str:
        smiles1 = canonicalize_molecule_smiles(smiles1)
        smiles2 = canonicalize_molecule_smiles(smiles2)
        if smiles1 is None and smiles2 is None:
            raise ChemAgentInputError("Both of the input molecules are invalid.")
        elif smiles1 is None:
            raise ChemAgentInputError('The first molecule is invalid.')
        elif smiles2 is None:
            raise ChemAgentInputError('The second molecule is invalid.')
        else:
            id1 = get_molecule_id(smiles1, remove_duplicate=False)
            id2 = get_molecule_id(smiles2, remove_duplicate=False)
            if id1 == id2:
                return 'identical'
            else:
                return 'different'


class CanonicalizeSMILES(BaseTool):
    name = "CanonicalizeSMILES"
    func_name = 'canonicalize_smiles'
    description = "Canonicalize SMILES representation. Input SMILES, returns canonicalized SMILES. You should use this tool when asked for canonicalized SMILES."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'OCC', 'output': 'CCO'},
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        if not is_smiles(smiles):
            raise ChemAgentInputError("Invalid SMILES.")
        
        smiles = canonicalize_molecule_smiles(smiles)
        if smiles is None:
            raise ChemAgentInputError("Invalid SMILES.")
        
        return smiles
    

class CountMolAtoms(BaseTool):
    name = "CountMolAtoms"
    func_name = 'count_molecule_atoms'
    description = "Count the number of atoms in a molecule. Input SMILES, returns the types of atoms and their numbers."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CCO', 'output': 'There are altogether 3 atoms (omitting hydrogen atoms). The types and corresponding numbers are: {"C": 2, "O": 1}'},
    ]

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return "Error: Invalid SMILES string"
        mol = Chem.rdmolops.AddHs(mol, explicitOnly=True)
        # Count the number of atoms
        num_atoms = mol.GetNumAtoms()
        # Get the atom types
        atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # Count the occurrences of each atom type
        atom_type_counts = {atom: atom_types.count(atom) for atom in set(atom_types)}
        
        text = "There are altogether %d atoms (omitting hydrogen atoms). The types and corresponding numbers are: %s" % (num_atoms, str(atom_type_counts))
        return text
