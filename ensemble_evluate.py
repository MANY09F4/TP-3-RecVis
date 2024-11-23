import argparse
import os
import numpy as np

import PIL.Image as Image
import torch
from tqdm import tqdm

from model_factory import ModelFactory

def opts() -> argparse.ArgumentParser:
    """Option Handling Function."""
    parser = argparse.ArgumentParser(description="RecVis A3 evaluation script")
    parser.add_argument(
        "--data",
        type=str,
        default="data_sketches",
        metavar="D",
        help="folder where data is located. test_images/ need to be found in the folder",
    )
    parser.add_argument(
        "--models",
        nargs='+',
        metavar="M",
        help="list of model files to be evaluated. Usually it is of the form model_X.pth",
    )
    parser.add_argument(
        "--model_names",
        nargs='+',
        default=["basic_cnn"],
        metavar="MOD",
        help="List of model names for model and transform instantiation.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="experiment/kaggle.csv",
        metavar="D",
        help="name of the output csv file",
    )
    parser.add_argument(
        "--data_augment",
        type=bool,
        default=False,
        metavar="DA",
        help="Data augmentation for the training sample",
    )
    parser.add_argument(
        "--ensemble",
        type=str,
        default="average",
        metavar="EN",
        help="Type of ensemble methods",
    )
    args = parser.parse_args()
    return args

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def main() -> None:
    """Main Function."""
    args = opts()
    test_dir = args.data + "/test_images/mistery_category"

    # cuda
    use_cuda = torch.cuda.is_available()

    # load models and transform
    models = []
    for model_path, model_name in zip(args.models, args.model_names):
        state_dict = torch.load(model_path)
        model, data_transforms = ModelFactory(model_name=model_name, test_mode=True).get_all()
        model.load_state_dict(state_dict)
        model.eval()
        if use_cuda:
            model.cuda()
        models.append(model)

    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    for f in tqdm(os.listdir(test_dir)):
        if "jpeg" in f:
            data = data_transforms(pil_loader(test_dir + "/" + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()

            if args.ensemble == "average" :
                # Get average prediction from all models
                avg_output = None
                for model in models:
                    output = model(data)
                    if avg_output is None:
                        avg_output = output
                    else:
                        avg_output += output
                avg_output /= len(models)

                pred = avg_output.data.max(1, keepdim=True)[1]
            elif args.ensemble == "maximum":
                outputs = []
                for model in models:
                    output = model(data)
                    outputs.append(output)
                stacked_outputs = torch.stack(outputs)
                max_output = torch.max(stacked_outputs, dim=0)[0]
                pred = torch.argmax(max_output, dim=1)

            output_file.write("%s,%d\n" % (f[:-5], pred.item()))

    output_file.close()

    print("Succesfully wrote " + args.outfile + ", you can upload this file to the kaggle competition website")

if __name__ == "__main__":
    main()
