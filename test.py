import glob
import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_builder import build_model
from builders.dataset_builder import build_dataset_test


def parse_args():
    parser = ArgumentParser(description='PUNet and DDNet')
    parser.add_argument('--model', type=str, default="PUNet", help="model name: (default PUNet)")
    parser.add_argument('--dataRootDir', type=str,
                        default=r"dataset",
                        help="dataset dir")
    parser.add_argument('--dataset', default="phaseUnwrapping", help="dataset")
    parser.add_argument('--num_workers', type=int, default=8, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing")
    parser.add_argument('--checkpoint', type=str,default="",
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    args = parser.parse_args()

    return args




def test(args, test_loader, model):
    """
    args:
      test_loader: loaded for test dataset
      model: model
    return: rmses
    """

    savePath = os.path.join(args.dataRootDir,args.dataset, args.model)
    os.makedirs(savePath,exist_ok=True)

    # evaluation or test mode
    model.eval()
    total_batches = len(test_loader)

    rmses = 0
    for i, (input, label, size, name) in enumerate(test_loader):
        with torch.no_grad():
            input_var = input.cuda()
        saveFile = os.path.join(savePath, name[0])
        start_time = time.time()
        output = model(input_var)
        torch.cuda.synchronize()
        time_taken = time.time() - start_time
        print('[%d/%d]  time: %.2f' % (i + 1, total_batches, time_taken))
        output = output.cpu().data[0].numpy()
        output.tofile(saveFile)

        gt = np.asarray(label[0].numpy(), dtype=np.float32)
        rmses += np.sqrt(np.mean((output - gt) ** 2))
        print(name)

    return rmses/(i+1)


def test_model(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    print(args)

    if args.cuda:
        print("=====> use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    # build the model
    model = build_model(args.model, num_channels=1)

    if args.cuda:
        model = model.cuda()  # using GPU for inference
        cudnn.benchmark = True

    # load the test set
    datas, testLoader = build_dataset_test(args.dataRootDir, args.dataset, args.num_workers,False)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print("=====> loading checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model'])
        else:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))
    else:
        try:
            p = sorted(glob.glob(os.path.join('checkpoint', args.dataset, args.model + 'bs*', '*.pth')))[-1]
            print("=====> loading checkpoint '{}'".format(p))
            checkpoint = torch.load(p)
            model.load_state_dict(checkpoint['model'])
            print("=====> loaded checkpoint '{}' (epoch {})".format(p, checkpoint['epoch']))
        except:
            print("=====> no checkpoint found at '{}'".format(args.checkpoint))

    print("=====> beginning validation")
    print("validation set length: ", len(testLoader))
    rmse = test(args, testLoader, model)
    print('rmse:',rmse)


if __name__ == '__main__':

    args = parse_args()
    test_model(args)