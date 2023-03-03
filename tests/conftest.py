from src.easyhla import EasyHLA


def make_comparison(easyhla: EasyHLA, ref_seq: str, test_seq: str) -> str:
    """
    Compares two sequences for differences

    :param easyhla: EasyHLA object
    :type easyhla: EasyHLA
    :param ref_seq: ...
    :type ref_seq: str
    :param test_seq: ...
    :type test_seq: str
    :return: A sequence where mismatches are replaced with '_'
    :rtype: str
    """
    ref, test = easyhla.nuc2bin(ref_seq.strip()), easyhla.nuc2bin(test_seq.strip())
    masked_seq = []

    for i in range(max(len(ref), len(test))):
        _r = ref[i] if i < len(ref) else 0
        _t = test[i] if i < len(test) else 0

        masked_seq.append(_r & _t)

    if len(ref) != len(test):
        if len(ref) > len(test):
            side_is_short = "test"
        elif len(ref) < len(test):
            side_is_short = "ref"
        return easyhla.bin2nuc(masked_seq) + f" [{side_is_short} is short]"
    return easyhla.bin2nuc(masked_seq)
