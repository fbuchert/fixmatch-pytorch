import logging


logger = logging.getLogger()


class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
