
class MutableParams:
    """
    mutable params. contains required and optional params
    """
    required = ["use_cuda", "local_rank", "do_train", "data_dir", "log_freq"]
    optional = []

    all_prams = set()

    def __init__(self, optional: list = None):
        if optional:
            self.optional = optional
            self.all_prams = set(self.required).union(set(self.optional))


if __name__ == '__main__':
    base = ["use_cuda", "local_rank", "do_train", "data_dir", "log_freq"]
    extra = ["x", "y", "z", "local_rank"]
    param = MutableParams(extra)
    assert param.all_prams == set(base).union(extra)
