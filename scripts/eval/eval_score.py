from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse

parser = argparse.ArgumentParser(description="Evaluate captioning results on COCO dataset.")
parser.add_argument("--annFile", type=str, default="./coco2014/annotations/captions_val2017.json", help="Path to the annotations JSON file.")
parser.add_argument("--resFile", type=str, required=True, help="Path to the results JSON file.")
args = parser.parse_args()



annFile = args.annFile
resFile = args.resFile

coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

cocoEval = COCOEvalCap(coco, cocoRes)
cocoEval.evaluate()
print("\n",resFile)
for metric, score in cocoEval.eval.items():
    print(f"{metric}: {score:.3f}")