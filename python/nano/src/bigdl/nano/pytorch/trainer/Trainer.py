#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from logging import warning
from operator import xor
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch import nn
from torch.fx.graph_module import GraphModule
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import _LRScheduler
from bigdl.nano.common import check_avx512
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnPlugin
from bigdl.nano.deps.ray.ray_api import distributed_ray
from bigdl.nano.deps.ipex.ipex_api import create_IPEXAccelerator, ipex_device
from bigdl.nano.deps.openvino.openvino_api import bind_openvino_methods
from bigdl.nano.deps.onnxruntime.onnxruntime_api import bind_onnxrt_methods


distributed_backends = ["spawn", "ray", "subprocess"]


class Trainer(pl.Trainer):

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16=False,
                 distributed_backend="spawn",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 *args: Any, **kwargs: Any) -> None:
        """
        A pytorch lightning trainer that uses bigdl-nano optimization.
        :param num_processes: number of processes in distributed training. default: 4.
        :param use_ipex: whether we use ipex as accelerator for trainer. default: True.
        :param cpu_for_each_process: A list of length `num_processes`, each containing a list of
            indices of cpus each process will be using. default: None, and the cpu will be
            automatically and evenly distributed among processes.
        """

        # Check keyword arguments
        if "accelerator" in kwargs:
            warning(f"""Accelerator will be specified by bigdl-nano,
            accelerator entered {kwargs['accelerator']} will be ignored. """)

            kwargs.pop('accelerator')
        if "plugins" in kwargs:
            warning(f"""Plugins will be specified by bigdl-nano,
             plugines entered {kwargs['plugins']} will be ignored. """)

            kwargs.pop('plugins')
        if cpu_for_each_process is not None:
            if len(cpu_for_each_process) != num_processes:
                raise ValueError(f"The length of `cpu_for_each_process` ("
                                 f"{len(cpu_for_each_process)}) is not equal to the number of"
                                 f" processes {num_processes}.")

        # Initialize trainer
        if use_ipex and not check_avx512():
            warning("Enable ipex in a cpu instruction set"
                    " without avx512 may cause some random error."
                    "Fall back to cpu device.")
            use_ipex = False

        if num_processes == 1:
            accelerator = None
            if use_ipex:
                accelerator = create_IPEXAccelerator(enable_bf16=enable_bf16)
            super().__init__(accelerator=accelerator, *args, **kwargs)
        else:
            plugin = None
            assert distributed_backend in distributed_backends, \
                f"Distributed backends supported now are spawn and ray," \
                " but get {distributed_backend}."
            if distributed_backend == "spawn":
                if use_ipex:
                    device = ipex_device()
                else:
                    device = "cpu"
                plugin = DDPSpawnPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "subprocess":
                from bigdl.nano.pytorch.plugins.ddp_subprocess import DDPSubprocessPlugin
                if use_ipex:
                    import intel_pytorch_extension as ipex
                    device = ipex.DEVICE
                else:
                    device = "cpu"
                plugin = DDPSubprocessPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "ray":
                # Import RayPlugins may entangle with openmp even if it has not been used,
                # which leads to an unacceptably low performance.
                # So we import when we need.
                plugin = distributed_ray(num_workers=num_processes,  # type: ignore
                                         use_ipex=use_ipex,
                                         device=ipex_device())

            accelerator = None
            if use_ipex:
                accelerator = create_IPEXAccelerator(training_type_plugin=plugin,  # type: ignore
                                                     enable_bf16=enable_bf16)

            super().__init__(accelerator=accelerator,
                             plugins=[plugin], *args, **kwargs)

    @staticmethod
    def compile(model: nn.Module,
                loss: _Loss = None,
                optimizer: torch.optim.Optimizer = None,
                scheduler: _LRScheduler = None,
                metrics: List[Metric] = None,
                onnx: bool = False,
                quantize: bool = False,
                openvino: bool = False):
        """
        Construct a pytorch-lightning model. If model is already a pytorch-lightning model,
        return model. If model is pytorch model, construct a new pytorch-lightning module
        with model, loss and optimizer.

        :param model:       A model instance.
        :param loss:        Loss to construct pytorch-lightning model.
                            Should be None if model is instance of pl.LightningModule.
        :param optimizer:   Optimizer to construct pytorch-lightning model Should be None.
                            if model is instance of pl.LightningModule.
        :param metrics:     A list of torchmetrics to validate/test performance.
        :param onnx:        Indicates if onnxruntime support should be binded to the
                            returned model.
        :param quantize:    Indicates if quantization support should be binded to the
                            returned model.
        :return:            A LightningModule object.
        """
        assert isinstance(model, nn.Module), \
            "Model must be instance of nn.Module but got {}".format(model.__class__)

        pl_model = None
        if isinstance(model, pl.LightningModule):
            assert not (loss or optimizer), \
                "Loss and optimizer should be None if model is a pytorch-lightning model."
            pl_model = model
        else:
            pl_model = LightningModuleFromTorch(model, loss, optimizer, scheduler, metrics)
        assert not (onnx and openvino), "Only one of onnx and openvino can be True."
        assert not (openvino and quantize), "Quantization is not implemented for OpenVINO."
        if onnx:
            try:
                from bigdl.nano.pytorch.runtime_binding.base_inference import\
                    bind_base_inference_rt_methods
                pl_model = bind_onnxrt_methods(bind_base_inference_rt_methods(pl_model))
            except ImportError:
                raise RuntimeError("You should install onnx and onnxruntime to set `onnx=True`, "
                                   "or just set `onnx=False`.")
        elif openvino:
            return bind_openvino_methods(pl_model)
        if quantize:
            from bigdl.nano.pytorch.runtime_binding.quantization_inference import\
                bind_quantize_methods
            from bigdl.nano.pytorch.runtime_binding.base_inference import\
                bind_base_inference_rt_methods
            pl_model = bind_quantize_methods(bind_base_inference_rt_methods(pl_model), None)

        return pl_model

    def quantize(self, pl_model,  # remove the type requirement for type checking
                 calib_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 metric: Optional[Metric] = None,
                 backend='inc',
                 conf: Optional[str] = None,
                 framework='pytorch_fx',
                 approach='static',
                 tuning_strategy='bayesian',
                 accuracy_criterion: dict = None,
                 timeout=0,
                 max_trials=1,
                 return_pl=True
                 ):
        """
        Calibrate a Pytorch-Lightning model for post-training quantization.

        :param pl_model:       A Pytorch-Lightning model to be quantized.
        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.
                                    Required for static quantization.
        :param val_dataloader:      A torch.utils.data.dataloader.DataLoader object for evaluation.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param backend:             Only 'inc' is supported. Default: 'inc'.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, using default config.
        :param framework:   string or list, [{'pytorch'|'pytorch_fx'|'pytorch_ipex'},
                            {'onnxrt_integerops'|'onnxrt_qlinearops'}]. Default: 'pytorch_fx'.
                            Consistent with Intel Neural Compressor.
        :param approach:    'static' or 'dynamic'.
                            'static': post_training_static_quant,
                            'dynamic': post_training_dynamic_quant.
                            Default: 'static'.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better': True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, 'higher_is_better':False} means accuracy
                                    must be smaller than 0.99.
        :param timeout:     Tuning timeout (seconds). Default: 0,  which means early stop.
                            Combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: 1.
                            Combine with timeout field to decide when to exit.
                            "timeout=0, max_trials=1" means it will try quantization only once and
                            return satisfying best model.
        :param return_pl:   Decide which type to return. If set to True, a GraphModule will be
                            returned. If set to False, a pytorch lightning module will be returned.
        :return:            A GraphModule. If there is no model found, return None.
        """
        has_pytorch = False
        has_onnx = False
        for framework_item in framework:
            if "pytorch" in framework_item:
                if has_pytorch:
                    raise RuntimeError("You should choose one from "
                                       "{'pytorch'|'pytorch_fx'|'pytorch_ipex'}")
                has_pytorch = True
                continue
            if "onnx" in framework_item:
                if has_onnx:
                    raise RuntimeError("You should choose one from "
                                       "{'onnxrt_integerops'|'onnxrt_qlinearops'}")
                has_onnx = True
                continue

        if backend == 'inc':
            from bigdl.nano.deps.neural_compressor.inc_api import QuantizationINC,\
                check_pytorch_dataloaders

            # check if dataloader is of legal format
            check_pytorch_dataloaders(pl_model, [calib_dataloader, val_dataloader])

            if approach not in ['static', 'dynamic']:
                raise ValueError("Approach should be 'static' or 'dynamic', "
                                 "{} is invalid.".format(approach))
            approach_map = {
                'static': 'post_training_static_quant',
                'dynamic': 'post_training_dynamic_quant'
            }
            approach = approach_map.get(approach)

            framework = [framework] if isinstance(framework, str) else framework
            quantized_models = []
            for framework_item in framework:
                quantizer = QuantizationINC(framework=framework_item, conf=conf, approach=approach,
                                            tuning_strategy=tuning_strategy,
                                            accuracy_criterion=accuracy_criterion,
                                            timeout=timeout, max_trials=max_trials)
                if "pytorch" in framework_item:
                    # for 'pytorch'|'pytorch_fx'|'pytorch_ipex'
                    model: Any = pl_model  # state model to be 'Any' since we may have pl or onnx
                    if isinstance(pl_model, LightningModuleFromTorch):
                        # LightningModuleFromTorch.forward fails to trace in FX
                        # so replace it temporarily
                        model = pl_model.model
                else:
                    # for 'onnxrt_integerops'|'onnxrt_qlinearops'
                    pl_model = bind_onnxrt_methods(pl_model)
                    if approach == "post_training_static_quant":
                        assert calib_dataloader, \
                            "calib_calib_dataloader must not be None when approach is " \
                            "post-training static quantization."
                        pl_model.eval_onnx(input_sample=tuple(next(iter(
                            calib_dataloader))[:-1]),
                            file_path="model.onnx")
                        model = pl_model.ort_infer_engine.onnx_model_fp32
                    else:
                        assert pl_model.ort_infer_engine.onnx_model_fp32, \
                            "Please call `eval_onnx` on model to " \
                            "update/build your onnx structure."
                        model = pl_model.ort_infer_engine.onnx_model_fp32
                quantized_models.append(quantizer.post_training_quantize(model, calib_dataloader,
                                                                         val_dataloader, metric))
            if not return_pl:
                if len(quantized_models) == 1:
                    return quantized_models[0].model
                else:
                    return quantized_models[0].model, quantized_models[1].model
            else:
                quantized_pytorch_model = None
                quantized_onnx_model = None
                for i, framework_item in enumerate(framework):
                    if "pytorch" in framework_item:
                        quantized_pytorch_model = quantized_models[i]
                    else:
                        quantized_onnx_model = quantized_models[i]
                from bigdl.nano.pytorch.runtime_binding.base_inference import \
                    bind_base_inference_rt_methods
                from bigdl.nano.pytorch.runtime_binding.quantization_inference import \
                    bind_quantize_methods
                return bind_onnxrt_methods(
                    bind_quantize_methods(
                        bind_base_inference_rt_methods(pl_model), quantized_pytorch_model),
                    quantized_onnx_model)
        else:
            raise NotImplementedError("Backend {} is not implemented.".format(backend))
