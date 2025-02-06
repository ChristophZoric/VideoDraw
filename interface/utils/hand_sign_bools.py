def isNeutralSign(hand_sig) -> bool:
    if hand_sig == 0:
        return True
    else:
        return False


def isDrawSign(hand_sig) -> bool:
    if hand_sig == 1:
        return True
    else:
        return False


def isDeletLastSign(hand_sig, annotations, frame_counter, frame_skip) -> bool:
    if hand_sig == 2 and annotations and frame_counter % frame_skip == 0:
        return True
    else:
        return False


def isDeletAllSign(hand_sig, annotations, frame_counter, frame_skip) -> bool:
    if hand_sig == 3 and annotations and frame_counter % frame_skip == 0:
        return True
    else:
        return False
