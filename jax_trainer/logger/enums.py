from enum import IntEnum


class LogMetricMode(IntEnum):
    """
    Enum for how to aggregate the metrics over the epochs.

    Attributes:
        MEAN:   average over the epochs
        SUM:    sum over the epochs
        SINGLE: only the last epoch is logged
        MAX:    maximum over the epochs
        MIN:    minimum over the epochs
        STD:    standard deviation over the epochs
    """

    MEAN = 1
    SUM = 2
    SINGLE = 3
    MAX = 4
    MIN = 5
    STD = 6


class LogStage(IntEnum):
    """
    Enum for when to log the metrics.

    Attributes:
        ANY:    log in any logging mode
        TRAIN:  log only during training
        VAL:    log only during validation
        TEST:   log only during testing
        EVAL:   log only during evaluation

    """

    ANY = 0
    TRAIN = 1
    VAL = 2
    TEST = 3
    EVAL = 4


class LogFreq(IntEnum):
    """
    Enum for how often to log the metrics.

    ANY:    log in any logging frequency
    STEP:   log every N step end
    EPOCH:  log every epoch end
    """

    ANY = 0
    STEP = 1
    EPOCH = 2
