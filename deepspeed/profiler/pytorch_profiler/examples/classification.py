import argparse
import sys
import torch
import torchvision.models as models
from deepspeed.profiler.pytorch_profiler import get_model_profile

pt_models = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'inception': models.inception_v3
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pytorch-profiler example script')
    parser.add_argument('--device',
                        type=int,
                        default=0,
                        help='Device to store the model.')
    parser.add_argument('--model',
                        choices=list(pt_models.keys()),
                        type=str,
                        default='resnet18')
    args = parser.parse_args()

    net = pt_models[args.model]()

    if torch.cuda.is_available():
        net.cuda(device=args.device)

    macs, params, steps = get_model_profile(
        net,
        (1, 3, 224, 224),
        print_profile=True,
        print_aggregated_profile=True,
        depth=-1,
        top_num=3,
        warm_up=5,
        num_steps=10,
        as_strings=True,
        ignore_modules=None)

    print('{:<30}  {:<8}'.format('Number of MACs: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print('{:<30}  {:<8}'.format('Number of steps profiled: ', steps))