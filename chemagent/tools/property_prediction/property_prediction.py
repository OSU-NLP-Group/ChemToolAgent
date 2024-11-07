import os
import logging

from ..base import BaseTool

logger = logging.getLogger(__name__)


from . import utils as pp_utils
from ...utils.smiles import is_smiles
from ...utils.error import *



MODEL_ARGS = {
    'esol': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/esol/checkpoint_best.pt',
        'task_name': 'esol',
        'task_num': 1,
        'loss_func': 'finetune_mse',
    },
    'lipo': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/lipo/checkpoint_best.pt',
        'task_name': 'lipo',
        'task_num': 1,
        'loss_func': 'finetune_mse',
    },
    'bbbp': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/bbbp/checkpoint_best.pt',
        'task_name': 'bbbp',
    },
    'clintox': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/clintox/checkpoint_best.pt',
        'task_name': 'clintox',
        'task_num': 1,
        'loss_func': 'multi_task_BCE',
    },
    'hiv': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/hiv/checkpoint_best.pt',
        'task_name': 'hiv',
    },
    'sider': {
        'model_loc': 'chemagent/tools/property_prediction/checkpoints/sider/checkpoint_best.pt',
        'task_name': 'sider',
        'task_num': 20,
        'loss_func': 'multi_task_BCE',
    },
}


class PropertyPredictor(BaseTool):
    def __init__(
        self, 
        task_name,
        init=True, 
        interface='text'
    ):
        self.task_name = task_name
        self.model = None
        self.task = None
        self.loss = None
        self.args = None
        super().__init__(init, interface=interface)

    def _init_modules(self):
        task_name = self.task_name
        cmd = pp_utils.construct_cmd(**MODEL_ARGS[task_name])
        self.args = pp_utils.parse_args(cmd)
        self.model, self.task, self.loss = pp_utils.load_model(self.args)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        if self.model is None or self.task is None or self.loss is None or self.args is None:
            self._init_modules()

        if not is_smiles(smiles):
            raise ChemAgentInputError(f"Invalid SMILES: {smiles}")
        task_num = 2
        if 'task_num' in MODEL_ARGS[self.task_name]:
            task_num = MODEL_ARGS[self.task_name]['task_num']
        loss_func = 'finetune_cross_entropy'
        if 'loss_func' in MODEL_ARGS[self.task_name]:
            loss_func = MODEL_ARGS[self.task_name]['loss_func']
        r = pp_utils.run_on_smiles(smiles, self.task_name, self.args, self.task, self.model, self.loss, task_num=task_num, loss_func=loss_func)
        return r


class PropertyPredictorESOL(PropertyPredictor):
    name = "SolubilityPredictor"
    func_name = 'cal_solubility'
    description = "Input SMILES of molecule/compound, returns the log solubility in mol/L."
    func_doc = ("smiles: str", "float")
    func_description = description
    examples = [
        {'input': 'CC(C)Cl', 'output': 'The log solubility in mol/L is -1.410.\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('esol', init, interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        return 'The log solubility in mol/L is {:.3f}.'.format(r) + '\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'


class PropertyPredictorLIPO(PropertyPredictor):
    name = "LogDPredictor"
    func_name = 'cal_logd'
    description = "Input SMILES of molecule/compound, returns the octanol/water distribution coefficient logD under the circumstance of pH 7.4."
    func_doc = ("smiles: str", "float")
    func_description = description
    examples = [
        {'input': 'NC(=O)C1=CC=CC=C1O', 'output': 'The octanol/water distribution coefficient logD under the circumstance of pH 7.4 is 1.090.\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('lipo', init, interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        return 'The octanol/water distribution coefficient logD under the circumstance of pH 7.4 is {:.3f}.'.format(r) + '\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'


class PropertyPredictorBBBP(PropertyPredictor):
    name = "BBBPPredictor"
    func_name = 'predict_bbbp'
    description = "Input SMILES of molecule/compound, returns the probability of the compound to penetrate the blood-brain barrier."
    func_doc = ("smiles: str", "float")
    func_description = description
    examples = [
        {'input': 'CCNC(=O)/C=C/C1=CC=CC(Br)=C1', 'output': 'The probability of the compound to penetrate the blood-brain barrier is 99.90%, which means it\'s likely to happen.\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('bbbp', init, interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        return 'The probability of the compound to penetrate the blood-brain barrier is {:.2f}%, which means it\'s {} to happen.'.format(r * 100, 'likely' if r >= 0.5 else 'unlikely') + '\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'


class PropertyPredictorClinTox(PropertyPredictor):
    name = "ToxicityPredictor"
    func_name = 'predict_toxicity'
    description = "Input SMILES of molecule/compound, returns the probability of the compound to be toxic."
    func_doc = ("smiles: str", "float")
    func_description = description
    examples = [
        {'input': 'COC[C@@H](NC(C)=O)C(=O)NCC1=CC=CC=C1', 'output': 'The probability of the compound to be toxic is 6.04%, which means it\'s unlikely to happen.\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('clintox', init,  interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        return 'The probability of the compound to be toxic is {:.2f}%, which means it\'s {} to happen.'.format(r * 100, 'likely' if r >= 0.5 else 'unlikely') + '\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'


class PropertyPredictorHIV(PropertyPredictor):
    name = "HIVInhibitorPredictor"
    func_name = 'predict_hiv'
    description = "Input SMILES of molecule/compound, returns the probability of the compound to be an inhibitor of HIV replication."
    func_doc = ("smiles: str", "float")
    func_description = description
    examples = [
        {'input': 'CC1=CN(C2C=CCCC2O)C(=O)NC1=O', 'output': 'The probability of the compound to be an inhibitor of HIV replication is 6.01%, which means it\'s unlikely to happen.\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('hiv', init, interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        return 'The probability of the compound to be an inhibitor of HIV replication is {:.2f}%, which means it\'s {} to happen.'.format(r * 100, 'likely' if r >= 0.5 else 'unlikely') + '\nNote that the result is predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'


class PropertyPredictorSIDER(PropertyPredictor):
    subtask_list = ['Blood and lymphatic system disorders', 'Cardiac disorders', 'Congenital, familial and genetic disorders', 'Ear and labyrinth disorders', 'Endocrine disorders', 'Eye disorders', 'Gastrointestinal disorders', 'Hepatobiliary disorders', 'Immune system disorders', 'Metabolism and nutrition disorders', 'Musculoskeletal and connective tissue disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)', 'Nervous system disorders', 'Pregnancy, puerperium and perinatal conditions', 'Psychiatric disorders', 'Renal and urinary disorders', 'Reproductive system and breast disorders', 'Respiratory, thoracic and mediastinal disorders', 'Skin and subcutaneous tissue disorders', 'Vascular disorders']

    name = "SideEffectPredictor"
    func_name = 'predict_side_effect'
    description = "Input SMILES of molecule/compound, returns the probabilities of the compound to cause different side effects. It can predict 20 different side effects, which are listed below: " + '; '.join(['(%d) %s' % (idx, subtask) for idx, subtask in enumerate(subtask_list, start=1)]) + "."
    func_doc = ("smiles: str", "str")
    func_description = description
    examples = [
        {'input': 'CC1=CC(C)=C(NC(=O)CN(CC(=O)O)CC(=O)O)C(C)=C1Br', 'output': "The probabilities of the compound to cause different side effects are as follows:Blood and lymphatic system disorders: 11.29%, which means it's unlikely to cause the side effect.\nCardiac disorders: 10.92%, which means it's unlikely to cause the side effect.\nCongenital, familial and genetic disorders: 11.98%, which means it's unlikely to cause the side effect.\nEar and labyrinth disorders: 8.48%, which means it's unlikely to cause the side effect.\nEndocrine disorders: 4.16%, which means it's unlikely to cause the side effect.\nEye disorders: 15.19%, which means it's unlikely to cause the side effect.\nGastrointestinal disorders: 57.00%, which means it's likely to cause the side effect.\nHepatobiliary disorders: 9.62%, which means it's unlikely to cause the side effect.\nImmune system disorders: 10.14%, which means it's unlikely to cause the side effect.\nMetabolism and nutrition disorders: 15.41%, which means it's unlikely to cause the side effect.\nMusculoskeletal and connective tissue disorders: 10.77%, which means it's unlikely to cause the side effect.\nNeoplasms benign, malignant and unspecified (incl cysts and polyps): 4.92%, which means it's unlikely to cause the side effect.\nNervous system disorders: 34.37%, which means it's unlikely to cause the side effect.\nPregnancy, puerperium and perinatal conditions: 3.32%, which means it's unlikely to cause the side effect.\nPsychiatric disorders: 8.06%, which means it's unlikely to cause the side effect.\nRenal and urinary disorders: 10.64%, which means it's unlikely to cause the side effect.\nReproductive system and breast disorders: 4.59%, which means it's unlikely to cause the side effect.\nRespiratory, thoracic and mediastinal disorders: 16.48%, which means it's unlikely to cause the side effect.\nSkin and subcutaneous tissue disorders: 53.97%, which means it's likely to cause the side effect.\nVascular disorders: 18.45%, which means it's unlikely to cause the side effect.\nNote that the results are predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed."},
    ]

    def __init__(
        self, 
        init=True, 
        interface='text'
    ):
        super().__init__('sider', init, interface=interface)

    def _run_base(self, smiles: str, *args, **kwargs) -> str:
        r = super()._run_base(smiles, *args, **kwargs)
        text = []
        for idx, prob in enumerate(r):
            prob = prob * 100
            text.append(f'{self.subtask_list[idx]}: {prob:.2f}%, which means it\'s {"likely" if prob >= 50 else "unlikely"} to cause the side effect.')
        description = 'The probabilities of the compound to cause different side effects are as follows:\n'
        text = description + '\n'.join(text)
        text += '\nNote that the results are predicted by a neural network model and may not be accurate. You may use other tools or resources to obtain more reliable results if needed.'
        return text
