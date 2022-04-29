import logging
import os
from collections import OrderedDict
import torch
from fvcore.nn.precise_bn import get_bn_modules
import weakref

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg, CfgNode
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    hooks,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer,
    TrainerBase
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from clearml import Task

from data import AugDatasetMapper
from models import resnet_IN_fpn


def add_custom_configs(cfg: CfgNode):
    _C = cfg

    _C.MODEL.INSTANCE_NORM = True

    _C.INPUT.LARGE_SCALE_JITTER = CfgNode()
    _C.INPUT.LARGE_SCALE_JITTER.ENABLED = True
    _C.INPUT.LARGE_SCALE_JITTER.MIN_SCALE = 0.2
    _C.INPUT.LARGE_SCALE_JITTER.MAX_SCALE = 2.0

    _C.SOLVER.PERIODIC_CHECKPOINTER = CfgNode({"ENABLED": True})
    _C.SOLVER.PERIODIC_CHECKPOINTER.PERIOD = _C.SOLVER.CHECKPOINT_PERIOD

    _C.SOLVER.BEST_CHECKPOINTER = CfgNode({"ENABLED": False})
    _C.SOLVER.BEST_CHECKPOINTER.METRIC = "bbox/AP50"
    _C.SOLVER.BEST_CHECKPOINTER.MODE = "max"

    _C.DATASETS.S3 = CfgNode()
    _C.DATASETS.S3.AWS_ENDPOINT_URL = ""
    _C.DATASETS.S3.AWS_ACCESS_KEY = ""
    _C.DATASETS.S3.AWS_SECRET_ACCESS = ""
    _C.DATASETS.S3.REGION_NAME = ""
    _C.DATASETS.S3.BUCKET = ""
    _C.DATASETS.S3.CERT_PATH = ""


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    def __init__(self, cfg, s3_info):
        """
        Args:
            cfg (CfgNode):
        """
        TrainerBase.__init__(self)
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg, s3_info)
        self.s3_info = s3_info

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg, self.s3_info),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if cfg.SOLVER.PERIODIC_CHECKPOINTER.ENABLED and comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.PERIODIC_CHECKPOINTER.PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model, self.s3_info)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if cfg.SOLVER.BEST_CHECKPOINTER and comm.is_main_process():
            ret.append(
                hooks.BestCheckpointer(
                    cfg.TEST.EVAL_PERIOD,
                    self.checkpointer,
                    cfg.SOLVER.BEST_CHECKPOINTER.METRIC,
                    mode=cfg.SOLVER.BEST_CHECKPOINTER.MODE
                )
            )

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, s3_info=None):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=AugDatasetMapper(cfg, s3_info, False)
        )

    @classmethod
    def build_train_loader(cls, cfg, s3_info=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(
            cfg, mapper=AugDatasetMapper(cfg, s3_info, True)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def test(cls, cfg, model, s3_info, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, s3_info)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info(
                    "Evaluation results for {} in csv format:".format(dataset_name)
                )
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args, cl_task=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_custom_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # if cl_task:
    #     cl_dict = cl_task.connect_configuration(name="hyperparams", configuration=cfg)
    #     cfg = CfgNode(init_dict=cl_dict)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args, cl_task_id=None):
    if cl_task_id is not None:
        cl_task = Task.current_task()
        assert (
            cl_task.task_id == cl_task_id
        ), "Current task in process does not match given task id!"
    else:
        cl_task = None

    if args.coco_dsnames:
        assert len(args.coco_dsnames) == len(args.coco_jsons)
        assert len(args.coco_dsnames) == len(args.coco_imgroots)
        for dsname, cjson, imroot in zip(
            args.coco_dsnames, args.coco_jsons, args.coco_imgroots
        ):
            register_coco_instances(dsname, {}, cjson, imroot)

    cfg = setup(args, cl_task)

    if args.s3_direct_read:
        s3_info = {
            "endpoint_url": cfg.DATASETS.S3.AWS_ENDPOINT_URL,
            "aws_access_key_id": cfg.DATASETS.S3.AWS_ACCESS_KEY,
            "aws_secret_access_key": cfg.DATASETS.S3.AWS_SECRET_ACCESS,
            "region_name": cfg.DATASETS.S3.REGION_NAME,
            "bucket": cfg.DATASETS.S3.BUCKET,
            "verify": cfg.DATASETS.S3.CERT_PATH if cfg.DATASETS.S3.CERT_PATH else None,
        }
    else:
        s3_info = None

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model, s3_info)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg, s3_info)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()
