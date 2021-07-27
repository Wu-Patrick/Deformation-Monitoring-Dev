import os
from torch.utils import data
from dataset.phaseUnwrapping import PhaseUnwrappingDataSet,PhaseUnwrappingValDataSet,PhaseUnwrappingTestDataSet

def build_dataset_train(dataRootDir,dataset, input_size, batch_size, random_mirror, num_workers):

    if dataset == 'phaseUnwrapping':
        data_dir = os.path.join(dataRootDir, dataset)
        train_data_list = os.path.join(data_dir, 'train.txt')
        val_data_list = os.path.join(data_dir, 'val.txt')

        trainLoader = data.DataLoader(
            PhaseUnwrappingDataSet(data_dir, train_data_list, crop_size=input_size,
                              mirror=random_mirror),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            pin_memory=True, drop_last=True)

        valLoader = data.DataLoader(
            PhaseUnwrappingValDataSet(data_dir, val_data_list),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
            drop_last=True)

        datas = None
        return datas, trainLoader, valLoader
    else:
        raise NotImplementedError(
            "not supports the dataset: %s" % dataset)

def build_dataset_test(dataRootDir, dataset, num_workers, none_gt=False):
    if dataset == 'phaseUnwrapping':
        data_dir = os.path.join(dataRootDir, dataset)
        test_data_list = os.path.join(data_dir, 'test.txt')
        if none_gt:
            testLoader = data.DataLoader(
                PhaseUnwrappingTestDataSet(data_dir, test_data_list),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        else:
            testLoader = data.DataLoader(
                PhaseUnwrappingValDataSet(data_dir, test_data_list),
                batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        datas = None
        return datas, testLoader

    else:
        raise NotImplementedError(
                "not supports the dataset: %s" % dataset)