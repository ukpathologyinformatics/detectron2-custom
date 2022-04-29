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

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
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
	setup_logger()

	
	# register_coco_instances("lesion_train", {}, "./datasets/lesion/train_labels.json", "./datasets/lesion/data/train")
	# register_coco_instances("lesion_val", {}, "./datasets/lesion/val_labels.json", "./datasets/lesion/data/val")
	# register_coco_instances("lesion_test", {}, "./datasets/lesion/test_labels.json", "./datasets/lesion/data/test")

	register_coco_instances("lesion_train", {}, "./datasets/lesion_scaled/train_labels.json", "./datasets/lesion_scaled/lesion_data_coco/train/data")
	register_coco_instances("lesion_val", {}, "./datasets/lesion_scaled/val_labels.json", "./datasets/lesion_scaled/lesion_data_coco/val/data")
	register_coco_instances("lesion_test", {}, "./datasets/lesion_scaled/test_labels.json", "./datasets/lesion_scaled/lesion_data_coco/test/data")


	# # visualize the data
	# dataset_dicts = DatasetCatalog.get('lesion_train')
	# for d in random.sample(dataset_dicts, 3):
	# 	img = cv2.imread(d["file_name"])
	# 	visualizer = Visualizer(img[:, :, ::-1], MetadataCatalog.get("lesion_train"))
	# 	out = visualizer.draw_dataset_dict(d)
	# 	cv2.imshow('train', out.get_image()[:, :, ::-1])
	# 	cv2.waitKey(0)


	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
	#cfg.CUDNN_BENCHMARK = True
	#cfg.SOLVER.WARMUP_ITERS = 1000
	cfg.DATASETS.TRAIN = ("lesion_train",)
	cfg.DATASETS.TEST = ("lesion_val", ) # dont put test dataset here
	cfg.DATALOADER.NUM_WORKERS = 8
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
	cfg.MODEL.DEVICE = 'cuda:0'
	cfg.SOLVER.IMS_PER_BATCH = 2
	cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
	cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
	cfg.SOLVER.STEPS = []        # do not decay learning rate
	cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
	cfg.TEST.EVAL_PERIOD = 500
	# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.



	os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
	#trainer = DefaultTrainer(cfg)
	trainer = CustomTrainer(cfg)
	trainer.resume_or_load(resume=False)
	trainer.train()


	# this is test
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
	predictor = DefaultPredictor(cfg)

	
	dataset_dicts = DatasetCatalog.get("lesion_test")
	for d in random.sample(dataset_dicts, 3):    
		im = cv2.imread(d["file_name"])
		outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
		v = Visualizer(im[:, :, ::-1],
					metadata=MetadataCatalog.get("lesion_test"), 
					scale=1, 
					instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
		)
		out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		cv2.imshow('test', out.get_image()[:, :, ::-1])
		cv2.waitKey(0)
		cv2.imwrite('./output/images/'+str(time.time()) + ".png", out.get_image()[:, :, ::-1])


	experiment_metrics = load_json_arr('./output/metrics.json')

	plt.plot(
		[x['iteration'] for x in experiment_metrics], 
		[x['total_loss'] for x in experiment_metrics])
	plt.plot(
		[x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
		[x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
	plt.legend(['total_loss', 'validation_loss'], loc='upper left')
	plt.show()


	# #test
	# from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
	# from detectron2.evaluation import COCOEvaluator, inference_on_dataset

	# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
	# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
	# predictor = DefaultPredictor(cfg)
	# evaluator = COCOEvaluator("lesion_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
	# val_loader = build_detection_test_loader(cfg, "lesion_test")
	# inference_on_dataset(trainer.model, val_loader, evaluator)