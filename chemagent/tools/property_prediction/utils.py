import os
import shutil
import os
import contextlib
import pickle
import lmdb
import hashlib
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import logging
import torch
import time

from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore.logging import progress_bar
from unicore import tasks


logger = logging.getLogger(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))



def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content):
    smi = content[0]
    target = content[1:]
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        coordinate_list = smi2_3Dcoords(smi,cnt)
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({'atoms': atoms, 
    'coordinates': coordinate_list, 
    'mol':mol,'smi': smi, 'target': target}, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None

def write_lmdb(smiles, outpath='./', ntargets=1):
    os.makedirs(outpath, exist_ok=True)
    output_name = os.path.join(outpath, "test.lmdb")
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    content = [smiles]
    for t in range(ntargets):
        content.append(0)
    inner_output = smi2coords(content)
    if inner_output is not None:
        txn_write.put(f'{i}'.encode("ascii"), inner_output)
        i += 1
    txn_write.commit()
    env_new.close()


def get_results_prob(predict_path, task=1):
    predict = pd.read_pickle(predict_path)
    smi_list, predict_list = [], []
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            predict_list.append(batch["prob"][i][task].cpu().tolist())
    predict_df = pd.DataFrame({"SMILES": smi_list, "predict_prob": predict_list})
    predict_df = predict_df.groupby("SMILES")["predict_prob"].mean().reset_index()
    return predict_df.iloc[0]["predict_prob"]


def get_results_predict(predict_path, task=1):
    predict = pd.read_pickle(predict_path)
    smi_list, predict_list = [], []
    for batch in predict:
        sz = batch["bsz"]
        for i in range(sz):
            smi_list.append(batch["smi_name"][i])
            predict_list.append(batch["predict"][i][task].cpu().tolist())
    predict_df = pd.DataFrame({"SMILES": smi_list, "predict_prob": predict_list})
    predict_df = predict_df.groupby("SMILES")["predict_prob"].mean().reset_index()
    return predict_df.iloc[0]["predict_prob"]


def get_results(predict_path, loss_func, task=1):
    if loss_func in ('multi_task_BCE', 'finetune_cross_entropy'):
        return get_results_prob(predict_path, task)
    elif loss_func == 'finetune_mse':
        return get_results_predict(predict_path, task)
    else:
        raise NotImplementedError(f"loss function {loss_func} not implemented")


def construct_cmd(
        model_loc, task_name, conf_size=11, only_polar=-1, task_num=2, loss_func="finetune_cross_entropy"
):
    unimol_location = os.path.join(dir_path, 'unimol')
    tmp_data = os.path.join(dir_path, 'data')
    cmd = f"{tmp_data} --user-dir {unimol_location} --task-name {task_name} --valid-subset test" \
            f" --num-workers 8 --ddp-backend=c10d --batch-size 1 --task mol_finetune --loss {loss_func} --arch unimol_base" \
            f" --classification-head-name {task_name} --num-classes {task_num} --dict-name dict.txt --conf-size {conf_size} --only-polar {only_polar}" \
            f" --path {model_loc} --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --log-interval 50 --log-format simple"
    
    return cmd


def parse_args(cmd):
    cmd = cmd.split()
    parser = options.get_validation_parser()
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser, cmd)
    return args


def load_model(args):
    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    # Load model
    logger.info("loading model(s) from {}".format(args.path))
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    # Move models to GPU
    if use_cuda:
        model.cuda()
        # fp16 only supported on CUDA for fused kernels
        if use_fp16:
            model.half()

    loss = task.build_loss(args)
    loss.eval()

    return model, task, loss


def run_on_smiles(smiles, task_name, args, task, model, loss, task_num=2, loss_func="finetune_cross_entropy"):
    smihash = hashlib.md5(smiles.encode()).hexdigest() + '_' + str(time.time())
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    parent_dir = os.path.join(dir_path, 'tmp_data')
    tmpdir = os.path.join(parent_dir, f"tmp_{smihash}")
    results_dir = os.path.join(parent_dir, f"results_{smihash}")
    try:
        os.makedirs(tmpdir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        args.data = tmpdir
        args.results_path = results_dir
        os.makedirs(os.path.join(tmpdir, task_name), exist_ok=True)
        shutil.copy(os.path.join(dir_path, 'data/dict.txt'), tmpdir)
        nt = task_num
        if nt == 2:
            nt = 1
        write_lmdb(smiles, os.path.join(tmpdir, task_name), nt)
        
        run_on_dataset(args, task, model, loss)

        predict_path = os.path.join(results_dir, f"{task_name}_test.out.pkl")

        if task_num == 2:
            return get_results(predict_path, loss_func, 1)
        elif task_num == 1:
            return get_results(predict_path, loss_func, 0)
        elif task_num > 2:
            return [get_results(predict_path, loss_func, j) for j in range(task_num)]
    finally:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)


def run_on_dataset(args, task, model, loss):
    use_cuda = torch.cuda.is_available() and not args.cpu
    if args.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)
        try:
            fname = (args.path).split("/")[-2]
        except:
            fname = 'infer'
        save_path = os.path.join(args.results_path, fname + "_" + subset + ".out.pkl")
        logger.info("Dataset Length: %d" % (len(dataset),))
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            batch_size=args.batch_size,
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )
        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if len(sample) == 0:
                continue
            _, _, log_output = task.valid_step(sample, model, loss, test=True)
            progress.log({}, step=i)
            log_outputs.append(log_output)
        pickle.dump(log_outputs, open(save_path, "wb"))
    return None
                
