import torch


class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]
