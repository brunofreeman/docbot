PROGRESS_BAR_SEGS: int = 100
PROGRESS_CHAR: str = "*"

def print_progress_bar(r: int, n_runs: int, label: str="") -> None:
    completion: float = r / n_runs
    completed_bars: int = round(completion * PROGRESS_BAR_SEGS)
    print("\r{}[{:3d}%] [{}{}]".format(
        f"{label}: " if label else "",
        round(100 * completion),
        PROGRESS_CHAR * completed_bars,
        " " * (PROGRESS_BAR_SEGS - completed_bars)
    ), end="", flush=True)


def print_progress_bar_complete(label: str="") -> None:
    print("\r{}[100%] [{}]".format(
        f"{label}: ".format(label) if label else "",
        PROGRESS_CHAR * PROGRESS_BAR_SEGS
    ))
