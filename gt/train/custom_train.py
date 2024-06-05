import logging
import time
from os.path import join

import numpy as np
import torch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt, get_ckpt_epoch, get_ckpt_path
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch

from gt.loss.subtoken_prediction_loss import subtoken_cross_entropy
from gt.utils import cfg_to_dict, flatten_dict, make_wandb_name, mlflow_log_cfgdict
import warnings
from collections import defaultdict
import rdkit
from rdkit import Chem
import pandas as pd

def train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    for iter, batch in enumerate(loader):
        # ipdb.set_trace()
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        pred, true = model(batch)

        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()


@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        if cfg.gnn.head == 'inductive_edge':
            pred, true, extra_stats = model(batch)
        else:
            pred, true = model(batch)
            extra_stats = {}
        if cfg.dataset.name == 'ogbg-code2':
            loss, pred_score = subtoken_cross_entropy(pred, true)
            _true = true
            _pred = pred_score
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()



@register_train('custom')
def custom_train(loggers, loaders, model, optimizer, scheduler):
    """
    Customized training pipeline.
    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler
    """
    # loggers[0].tb_writer.add_graph(model.model)


    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler,
                                cfg.train.epoch_resume)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    if cfg.mlflow.use:
        try:
            import mlflow
        except:
            raise ImportError('MLflow is not installed.')
        if cfg.mlflow.name == '':
            MLFLOW_NAME = make_wandb_name(cfg)
        else:
            MLFLOW_NAME = cfg.mlflow.name

        if cfg.name_tag != '':
            MLFLOW_NAME = MLFLOW_NAME + '-' + cfg.name_tag

        experiment = mlflow.set_experiment(cfg.mlflow.project)
        mlflow.start_run(run_name=MLFLOW_NAME)
        mlflow.pytorch.log_model(model, "model")
        mlflow_log_cfgdict(cfg_to_dict(cfg), mlflow_func=mlflow)
        if cfg.get('cfg_file', None) is not None: mlflow.log_artifact(cfg.cfg_file) # log the whole config-file



    num_splits = len(loggers)
    split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]

    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        # with torch.autograd.detect_anomaly():
        start_time = time.perf_counter()
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                    cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        # debug = True
        # if debug:
        #     tb_writer = loggers[0].tb_writer
        #     for k,v in model.named_buffers():
        #         if "running" in k:
        #             tb_writer.add_text(k, str(v), global_step=cur_epoch)

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                try:
                    eval_epoch(loggers[i], loaders[i], model,
                               split=split_names[i - 1])
                    perf[i].append(loggers[i].write_epoch(cur_epoch))
                except Exception as e:
                    print(e)
                    perf[i].append(perf[i][-1])
                    # reset the running_mean and running_var for batchnorm is extreme results occurs
                    warnings.warn(f"NaN occurs on the {split_names[i - 1]}_loss, reset the running-mean and running-var for BN")
                    for k, v in model.named_buffers():
                        if "running_mean" in k:
                            v.data = torch.zeros_like(v.data)
                        if "running_var" in k:
                            v.data = torch.ones_like(v.data) * 1e3
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau' or "timm" in cfg.optim.scheduler:
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and not cfg.train.ckpt_best \
                and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf), step=cur_epoch)

        if cfg.mlflow.use:
            mlflow.log_metrics(flatten_dict(perf), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_epoch_loss = best_epoch

            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()

                # if cfg.get("mv_metric_best", False):
                #     mv_len = cfg.get("mv_len", 10)
                #     mv_metric = np.array([vp[m] for vp in val_perf])
                #     if len(mv_metric) > mv_len:
                #         mv_metric = np.array([np.mean(mv_metric[max(i-mv_len, 0):i+1]) for i in range(len(mv_metric))])
                #     best_epoch = getattr(mv_metric, cfg.metric_agg)()


                if cfg.best_by_loss:
                    best_epoch = best_epoch_loss

                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use or cfg.mlflow.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            if cfg.wandb.use:
                                run.summary[f"best_{s}_perf"] = \
                                    perf[i][best_epoch][m]
                            if cfg.mlflow.use:
                                mlflow.log_metric(f"best_{s}_perf", perf[i][best_epoch][m])

                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    if cfg.wandb.use:
                        run.log(bstats, step=cur_epoch)
                        run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                        run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)

                    if cfg.mlflow.use:
                        mlflow.log_metrics(bstats, step=cur_epoch)
                        mlflow.log_metric("full_epoch_time_avg", np.mean(full_epoch_times))
                        mlflow.log_metric("full_epoch_time_sum", np.sum(full_epoch_times))


            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == cur_epoch:
                if cur_epoch < cfg.optim.num_warmup_epochs:
                    pass
                else:
                    save_ckpt(model, optimizer, scheduler, cur_epoch)
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t" 
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}\n"
                f"-----------------------------------------------------------"
            )

        if cfg.optim.get('early_stop_by_lr', False):
            lr = scheduler.get_last_lr()[0]
            if lr <= cfg.optim.min_lr:
                break
        elif cfg.optim.get("early_stop_by_perf", False):
            if (cur_epoch - best_epoch) > cfg.optim.patience:
                break

    if cfg.train.enable_ckpt and cfg.train.ckpt_best and cfg.mlflow.use:
        mlflow.log_artifact(get_ckpt_path(get_ckpt_epoch(best_epoch)))

    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")


    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    if cfg.mlflow.use:
        mlflow.end_run()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)

@register_train('GTNMR-inference')
def gtnmr_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on GT-NMR dataset.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = 1
    assert len(loaders) == num_splits, "Expecting 1 particular split."

    model.eval()
    start_time = time.perf_counter()
    results_list = []
    for i in range(num_splits):
        for batch in loaders[i]:
            batch.to(torch.device(cfg.device))
            data_id = batch.data_id
            pred, true = model(batch)
            print(pred)
            print(data_id)
            print('-------------------')

            pred = pred.detach().cpu()
            #true = true.detach().cpu()
            out_dict = {'y_pred': pred, 'y_true': true, 'data_id': data_id}
            results_list.append(out_dict)
    # save the results
    save_file = join(cfg.run_dir, f"results_{cfg.dataset.name}.pickle")
    logging.info(f"Saving to file: {save_file}")
    results_list = [{k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in results_list]
    import pickle
    with open(save_file, 'wb') as f:
        pickle.dump(results_list, f)

    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')


@register_train('GTNMR-custom-inference')
def gtnmr_custom_inference(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference on GT-NMR dataset.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    from pathlib import Path
    import os
    num_splits = 1
    assert len(loaders) == num_splits, "Expecting 1 particular split."
    from gt.train.sscore import calculate_score_from_smiles
    model.eval()
    start_time = time.perf_counter()
    results_list = []
    # 确保输出目录存在
    output_dir = Path(__file__).resolve().parents[2] / 'inference_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mol_list = []
    name_list = []
    for i in range(num_splits):
        for batch in loaders[i]:
            batch.to(torch.device(cfg.device))
            pred, true = model(batch)
            mol = batch.mol
            mol = mol[0]
            mol_list.append(mol)
            name = batch.name
            name_list.append(name)
            rows = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'C':
                    index = atom.GetIdx()
                    index += 1
                    rows.append({"Atom": index})
            df1 = pd.DataFrame(rows)
            pred_df = pd.DataFrame(pred.detach().cpu().numpy(), columns=['Predicted 13C Chemical Shifts'])
            pred_df['Predicted 13C Chemical Shifts'] = pred_df['Predicted 13C Chemical Shifts'].apply(lambda x: round(x, 2))
            df = pd.concat([df1, pred_df], axis=1)

            # 打印 DataFrame 和 nSPS
            smiles = Chem.MolToSmiles(mol)
            sps = calculate_score_from_smiles(smiles)
            num_atoms = mol.GetNumAtoms()
            nSPS = sps / num_atoms
            output_text = ('=====================================================' + '\n' + f'name: {str(name[0])}\n' +
                    df.to_string(index=False) + '\n' + f'nSPS: {nSPS}\n')
            print(output_text)
            # 将结果添加到 results_list
            results_list.append({'out_txt': output_text})
    # 保存结果为 .txt 文件
    output_file = output_dir/ f'inference_results_{cfg.dataset.inference}.txt'
    with open(output_file, 'w') as f:
        for result in results_list:
            f.write(result['out_txt'] + '\n')
    for mol in mol_list:
        name = name_list[mol_list.index(mol)]
        name=str(name[0])
        mol_file = output_dir / f'{name}.mol'
        Chem.MolToMolFile(mol, str(mol_file))
    logging.info(f"Results saved to {output_file}")
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')


@register_train('inference-only')
def inference_only(loggers, loaders, model, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    num_splits = len(loggers)
    split_names = ['train', 'val', 'test']
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0
    start_time = time.perf_counter()

    for i in range(0, num_splits):
        eval_epoch(loggers[i], loaders[i], model,
                   split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))

    best_epoch = 0
    best_train = best_val = best_test = ""
    if cfg.metric_best != 'auto':
        # Select again based on val perf of `cfg.metric_best`.
        m = cfg.metric_best
        if m in perf[0][best_epoch]:
            best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
        else:
            # Note: For some datasets it is too expensive to compute
            # the main metric on the training set.
            best_train = f"train_{m}: {0:.4f}"
        best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
        best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

    logging.info(
        f"> Inference | "
        f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
        f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
        f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
    )
    logging.info(f'Done! took: {time.perf_counter() - start_time:.2f}s')
    for logger in loggers:
        logger.close()







