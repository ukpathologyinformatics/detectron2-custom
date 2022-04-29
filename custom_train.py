# ###############TUTORIAL!!!!!: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=YU5_W8wJF02F
# Some basic setup:
# Setup detectron2 logger
# from __future__ import annotations
import detectron2
from detectron2.utils.logger import setup_logger

# import some common libraries
import numpy as np
import os, json, cv2, random
from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from clearml import Task
from det2_default_argparser import default_argument_parser
from det2_helper import extend_opts,  parse_datasets_args

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer, launch
from trainer_clearml import main
from detectron2.data.datasets import register_coco_instances
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
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from LossEvalHook import *

def build_evaluator(cfg, dataset_name, output_folder=None):
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
		return CityscapesInstanceEvaluator(dataset_name)
	if evaluator_type == "cityscapes_sem_seg":
		return CityscapesSemSegEvaluator(dataset_name)
	elif evaluator_type == "pascal_voc":
		return PascalVOCDetectionEvaluator(dataset_name)
	elif evaluator_type == "lvis":
		return LVISEvaluator(dataset_name, output_dir=output_folder)
	if len(evaluator_list) == 0:
		raise NotImplementedError(
			"no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
		)
	elif len(evaluator_list) == 1:
		return evaluator_list[0]
	return DatasetEvaluators(evaluator_list)


class CustomTrainer(DefaultTrainer):
	"""
	We use the "DefaultTrainer" which contains pre-defined default logic for
	standard training workflow. They may not work for you, especially if you
	are working on a new research project. In that case you can write your
	own training loop. You can use "tools/plain_train_net.py" as an example.
	"""

	@classmethod
	def build_evaluator(cls, cfg, dataset_name, output_folder=None):
		return build_evaluator(cfg, dataset_name, output_folder)

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

def load_json_arr(json_path):
		lines = []
		with open(json_path, 'r') as f:
			for line in f:
				lines.append(json.loads(line))
		return lines


if __name__ == '__main__':

	# ClearML Stuff
	parser = default_argument_parser()
	parser.add_argument(
		"--skip-clearml",
		help="flag to entirely skip any clearml action.",
		action="store_true",
	)
	parser.add_argument(
		"--monitor-ps", help="flag to monitoring processes.", action="store_true"
	)
	parser.add_argument(
		"--clearml-run-locally",
		help="flag to run job locally but keep clearml expt tracking.",
		action="store_true",
	)
	## CLEARML ARGS
	parser.add_argument("--clearml-proj", default="Detectron2", help="ClearML Project Name")
	parser.add_argument("--clearml-task-name", default="Task", help="ClearML Task Name")
	parser.add_argument(
		"--clearml-task-type",
		default="data_processing",
		help="ClearML Task Type, e.g. training, testing, inference, etc",
		choices=[
			"training",
			"testing",
			"inference",
			"data_processing",
			"application",
			"monitor",
			"controller",
			"optimizer",
			"service",
			"qc",
			"custom",
		],
	)
	parser.add_argument(
		"--clearml-output-uri",
		# default="s3://ecs.dsta.ai:80/clearml-models/default",
		help="ClearML output uri",
	)
	parser.add_argument(
		"--docker-img",
		default="harbor.dsta.ai/nvidia/pytorch:21.03-py3",
		help="Base docker image to pull",
	)
	parser.add_argument("--queue", default="1gpu", help="ClearML Queue")
	### S3
	parser.add_argument(
		"--skip-s3", help="flag to entirely skip any s3 action.", action="store_true"
	)
	## DOWNLOAD MODELS ARGS
	parser.add_argument(
		"--download-models", help="List of models to download", nargs="+"
	)
	parser.add_argument("--s3-models-bucket", help="S3 Bucket for models")
	parser.add_argument("--s3-models-path", help="S3 Models Path")
	## Model weights to load for training
	parser.add_argument(
		"--model-weights", help="MODEL.WEIGHTS | Path to pretrained model weights"
	)
	parser.add_argument(
		"--from-scratch",
		help="MODEL.WEIGHTS set to empty string | Train model from scratch",
		action="store_true",
	)
	## DOWNLOAD DATA ARGS
	parser.add_argument(
		"--download-data", help="List of dataset to download", nargs="+"
	)
	parser.add_argument(
		"--local-data-dir", help="Destination dataset files downloaded to", default='datasets'
	)
	parser.add_argument("--s3-data-bucket", help="S3 Bucket for data")
	parser.add_argument("--s3-data-path", help="S3 Data Path")
	parser.add_argument(
		"--s3-direct-read",
		help="DATASETS.S3.ENABLED | enable direct reading of images from S3 bucket without initial download.",
		action="store_true",
	)
	parser.add_argument(
		"--coco-dsnames",
		help="Names of custom datasets (must match to those in args.datasets_train and args.datasets_test(val)).",
		nargs="*",
	)
	parser.add_argument(
		"--coco-jsons",
		help="Paths to coco json file.",
		nargs="*",
	)
	parser.add_argument(
		"--coco-imgroots",
		help="Paths to img roots.",
		nargs="*",
	)
	# Datasets to register, will override config.yaml
	parser.add_argument("--datasets-train", help="DATASETS.TRAIN")
	parser.add_argument("--datasets-test", help="DATASETS.TEST")
	# ## UPLOAD OUTPUT ARGS
	# parser.add_argument("--s3-output-bucket", help="S3 Bucket for output")
	# parser.add_argument("--s3-output-path", help="S3 Path to output")
	## Hyperparams
	parser.add_argument("--num-classes", help="MODEL.ROI_HEADS.NUM_CLASSES")
	parser.add_argument("--test-eval-period", help="TEST.EVAL_PERIOD")
	parser.add_argument("--solver-ims-per-batch", help="SOLVER.IMS_PER_BATCH")
	parser.add_argument("--solver-base-lr", help="SOLVER.BASE_LR")
	parser.add_argument("--solver-gamma", help="SOLVER.GAMMA")
	parser.add_argument("--solver-warmup-iters", help="SOLVER.WARMUP_ITERS")
	parser.add_argument("--solver-steps", help="SOLVER.STEPS")
	parser.add_argument("--solver-max-iter", help="SOLVER.MAX_ITER")
	parser.add_argument("--dataloader-num-workers", help="DATALOADER.NUM_WORKERS")
	parser.add_argument("--model-anchor-sizes", help="MODEL.ANCHOR_GENERATOR.SIZES")
	parser.add_argument(
		"--model-anchor-ar", help="MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS"
	)

	args = parser.parse_args()
	print("Command Line Args:", args)

	environs_names = ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY", "AWS_SECRET_ACCESS", "CERT_PATH", "CERT_DL_URL"]
	environs = {var: os.environ.get(var) for var in environs_names}

	"""
	clearml task init
	"""
	if not args.skip_clearml:
		cl_task = Task.init(
			project_name=args.clearml_proj,
			task_name=args.clearml_task_name,
			task_type=args.clearml_task_type,
			output_uri=args.clearml_output_uri,
		)
		env_strs = ' '.join([ f"--env {k}={v}" for k, v in environs.items() ])
		cl_task.set_base_docker(
			f"{args.docker_img} --env GIT_SSL_NO_VERIFY=true {env_strs}"
		)
		if not args.clearml_run_locally:
			cl_task.execute_remotely(queue_name=args.queue, exit_process=True)
		cl_task_id = cl_task.task_id
	else:
		cl_task = None
		cl_task_id = None

	"""
	S3 handling to download weights and datasets
	"""
	local_weight_dir = "weights"
	local_data_dir = args.local_data_dir
	local_output_dir = "output"

	if not args.skip_s3:
		from utils.s3_helper import S3_handler

		environs['CERT_PATH'] = environs['CERT_PATH'] if environs['CERT_PATH'] else None
		if environs['CERT_DL_URL'] and environs['CERT_PATH'] and not Path(environs['CERT_PATH']).is_file():
			import utils.wget as wget
			import ssl
			ssl._create_default_https_context = ssl._create_unverified_context
			print(f'Downloading from {environs["CERT_DL_URL"]}')
			wget.download(environs['CERT_DL_URL'])
			environs['CERT_PATH'] = Path(environs['CERT_DL_URL']).name

		s3_handler = S3_handler(
			environs['AWS_ENDPOINT_URL'], environs['AWS_ACCESS_KEY'], environs['AWS_SECRET_ACCESS'], environs['CERT_PATH']
		)

		if args.download_models:
			local_weights_paths = s3_handler.dl_files(
				args.download_models,
				args.s3_models_bucket,
				args.s3_models_path,
				local_weight_dir,
				unzip=True,
			)

		if args.download_data:
			if args.s3_direct_read:
				local_data_dirs = s3_handler.dl_files(
					args.download_data,
					args.s3_data_bucket,
					args.s3_data_path,
					local_data_dir,
					unzip=True,
				)
			else:
				local_data_dirs = s3_handler.dl_dirs(
					args.download_data,
					args.s3_data_bucket,
					args.s3_data_path,
					local_data_dir,
					unzip=True,
				)

	"""
	Datasets Registration
	"""
	datasets_to_reg = []
	datasets_train = parse_datasets_args(args.datasets_train, datasets_to_reg)
	datasets_test = parse_datasets_args(args.datasets_test, datasets_to_reg)

	extend_opts(args.opts, "DATASETS.TRAIN", datasets_train)
	extend_opts(args.opts, "DATASETS.TEST", datasets_test)
	if args.from_scratch:
		extend_opts(args.opts, "MODEL.WEIGHTS", "")
	else:
		extend_opts(args.opts, "MODEL.WEIGHTS", args.model_weights)

	extend_opts(args.opts, "MODEL.ROI_HEADS.NUM_CLASSES", args.num_classes)
	extend_opts(args.opts, "TEST.EVAL_PERIOD", args.test_eval_period)
	extend_opts(args.opts, "SOLVER.IMS_PER_BATCH", args.solver_ims_per_batch)
	extend_opts(args.opts, "SOLVER.BASE_LR", args.solver_base_lr)
	extend_opts(args.opts, "SOLVER.GAMMA", args.solver_gamma)
	extend_opts(args.opts, "SOLVER.WARMUP_ITERS", args.solver_warmup_iters)
	extend_opts(args.opts, "SOLVER.STEPS", args.solver_steps)
	extend_opts(args.opts, "SOLVER.MAX_ITER", args.solver_max_iter)
	extend_opts(args.opts, "DATALOADER.NUM_WORKERS", args.dataloader_num_workers)
	extend_opts(args.opts, "MODEL.ANCHOR_GENERATOR.SIZES", args.model_anchor_sizes)
	extend_opts(args.opts, "MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS", args.model_anchor_ar)

	if args.s3_direct_read:
		extend_opts(args.opts, "DATASETS.S3.AWS_ENDPOINT_URL", environs['AWS_ENDPOINT_URL'])
		extend_opts(args.opts, "DATASETS.S3.AWS_ACCESS_KEY", environs['AWS_ACCESS_KEY'])
		extend_opts(args.opts, "DATASETS.S3.AWS_SECRET_ACCESS", environs['AWS_SECRET_ACCESS'])
		extend_opts(args.opts, "DATASETS.S3.REGION_NAME", "us-east-1")
		extend_opts(args.opts, "DATASETS.S3.BUCKET", args.s3_data_bucket)
		extend_opts(args.opts, "DATASETS.S3.CERT_PATH", environs['CERT_PATH'])

	"""
	Launching detectron2 run
	"""
	if args.monitor_ps:
		from psutil_helper import start_monitor

		start_monitor(freq=1)

	launch(
		main,
		args.num_gpus,
		num_machines=args.num_machines,
		machine_rank=args.machine_rank,
		dist_url=args.dist_url,
		args=(args, cl_task_id),
	)

	"""
	S3 handling to upload outputs
	"""
	if args.eval_only:
		from tester_clearml import coco_eval

		pred_path = Path(local_output_dir)
		pred_path = pred_path / "inference/coco_instances_results.json"
		if cl_task:
			cl_task.upload_artifact(
				name="predictions",
				artifact_object=str(pred_path),
			)
		if args.custom_cocojsons:
			for custom_json in args.custom_cocojsons:
				custom_json_path = Path(custom_json)
				if "val" in custom_json_path.stem:
					data_dir = custom_json_path.parent
					break
		else:
			data_dir = Path(local_data_dir)
		datasets_test = (
			datasets_test[0] if isinstance(datasets_test, tuple) else datasets_test
		)
		evals = coco_eval(pred_path, data_dir, val_str="val", subfolder=datasets_test)
		if cl_task:
			cl_task.upload_artifact(
				name="evaluations",
				artifact_object=evals,
			)
			cl_logger = cl_task.get_logger()
			for val_set, eval_values in evals.items():
				for metric, value in eval_values.items():
					cl_logger.report_scalar(
						title=val_set.replace("_coco-catified", ""),
						series=metric,
						value=value,
						iteration=0,
					)






















	# setup_logger()

	
	# # register_coco_instances("lesion_train", {}, "./datasets/lesion/train_labels.json", "./datasets/lesion/data/train")
	# # register_coco_instances("lesion_val", {}, "./datasets/lesion/val_labels.json", "./datasets/lesion/data/val")
	# # register_coco_instances("lesion_test", {}, "./datasets/lesion/test_labels.json", "./datasets/lesion/data/test")

	# register_coco_instances("lesion_train", {}, "./datasets/lesion_scaled/train_labels.json", "./datasets/lesion_scaled/lesion_data_coco/train/data")
	# register_coco_instances("lesion_val", {}, "./datasets/lesion_scaled/val_labels.json", "./datasets/lesion_scaled/lesion_data_coco/val/data")
	# register_coco_instances("lesion_test", {}, "./datasets/lesion_scaled/test_labels.json", "./datasets/lesion_scaled/lesion_data_coco/test/data")


	# # # visualize the data
	# # dataset_dicts = DatasetCatalog.get('lesion_train')
	# # for d in random.sample(dataset_dicts, 3):
	# # 	img = cv2.imread(d["file_name"])
	# # 	visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get("lesion_train"))
	# # 	out = visualizer.draw_dataset_dict(d)
	# # 	cv2.imshow('train', out.get_image()[:, :, ::-1])
	# # 	cv2.waitKey(0)


	# cfg = get_cfg()
	# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
	# #cfg.CUDNN_BENCHMARK = True
	# #cfg.SOLVER.WARMUP_ITERS = 1000
	# cfg.DATASETS.TRAIN = ("lesion_train",)
	# cfg.DATASETS.TEST = ("lesion_val", ) # dont put test dataset here
	# cfg.DATALOADER.NUM_WORKERS = 8
	# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
	# cfg.MODEL.DEVICE = 'cuda:0'
	# cfg.SOLVER.IMS_PER_BATCH = 2
	# cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
	# cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
	# cfg.SOLVER.STEPS = []        # do not decay learning rate
	# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
	# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
	# cfg.TEST.EVAL_PERIOD = 500
	# # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.



	# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	# #trainer = DefaultTrainer(cfg)
	# trainer = CustomTrainer(cfg)
	# trainer.resume_or_load(resume=False)
	# trainer.train()


	# # this is test
	# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
	# predictor = DefaultPredictor(cfg)

	
	# dataset_dicts = DatasetCatalog.get("lesion_test")
	# for d in random.sample(dataset_dicts, 3):    
	# 	im = cv2.imread(d["file_name"])
	# 	outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
	# 	v = Visualizer(im[:, :, ::-1],
	# 				metadata=MetadataCatalog.get("lesion_test"), 
	# 				scale=1, 
	# 				instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
	# 	)
	# 	out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
	# 	cv2.imshow('test', out.get_image()[:, :, ::-1])
	# 	cv2.waitKey(0)
	# 	cv2.imwrite('./output/images/'+str(time.time()) + ".png", out.get_image()[:, :, ::-1])


	# experiment_metrics = load_json_arr('./output/metrics.json')

	# plt.plot(
	# 	[x['iteration'] for x in experiment_metrics], 
	# 	[x['total_loss'] for x in experiment_metrics])
	# plt.plot(
	# 	[x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
	# 	[x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
	# plt.legend(['total_loss', 'validation_loss'], loc='upper left')
	# plt.show()


	# #test
	# from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
	# from detectron2.evaluation import COCOEvaluator, inference_on_dataset

	# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
	# predictor = DefaultPredictor(cfg)
	# evaluator = COCOEvaluator("lesion_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
	# val_loader = build_detection_test_loader(cfg, "lesion_test")
	# inference_on_dataset(trainer.model, val_loader, evaluator)
