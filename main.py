from typing import Tuple, Sequence, Optional

import numpy as np
import matplotlib.pyplot as plt


def compute_frames_per_animation(
        attacks_per_second: float,
        base_animation_length: int,
        speed_coefficient: float = 1.0,
        engine_tick_rate: int = 60,
        is_channeling: bool = False) -> int:
    """Calculates frames per animation needed to resolve a certain ability at attacks_per_second.

    Args:
        attacks_per_second: attacks per second of character
        base_animation_length: animation length of ability
        speed_coefficient: speed-up scalar of ability
        engine_tick_rate: server tick rate
        is_channeling: whether or not the ability is a channeling skill

    Returns:
        int: number of frames one casts needs to resolve for
    """
    _coeff = engine_tick_rate / (attacks_per_second * speed_coefficient)
    if is_channeling:
        return np.floor(_coeff)
    else:
        return np.ceil((base_animation_length - 1) / base_animation_length * _coeff)


def compute_character_attack_speed(
        frames_per_animation: int,
        base_animation_length: int,
        speed_coefficient: float = 1.0,
        engine_tick_rate: int = 60) -> Tuple[float, float]:
    """Computes the minimum and maximum character_attack_speed for a certain ability to reach the specified frames
    per animation breakpoint.

    Args:
        frames_per_animation: breakpoint to calculate attack speed values for
        base_animation_length: animation length of ability
        speed_coefficient: speed-up scalar of ability
        engine_tick_rate: server tick rate

    Returns:
        Tuple[int, int]: min attack speed to reach frames_per_animation breakpoint and max attack speed to leave
        frames_per_animation breakpoint.
    """
    _coeff = (base_animation_length - 1) * engine_tick_rate / (speed_coefficient * base_animation_length)
    min_aps = _coeff / frames_per_animation
    max_aps = _coeff / (frames_per_animation - 1)
    return min_aps, max_aps


def compute_phantasm_ticks_per_frame(
        frames_per_animation: int,
        casting_delay_frames: int,
        phantasm_tick_rate: int = 60,
        num_phantasms: int = 3) -> float:
    """Computes 'ticks_per_frame' for Spirit Barrage Phantasms at frames_per_breakpoint when sniping for
    casting_delay_frames.
    Args:
        frames_per_animation: Spirit Barrage breakpoint
        casting_delay_frames: number of frames to snipe with after all num_phantasms have been placed
        phantasm_tick_rate: damage instance tick rate of phantasms
        num_phantasms: maximum number of phantasms

    Returns:
        float: how many ticks resolve per frame
    """
    casting_frames = frames_per_animation * num_phantasms + casting_delay_frames
    num_ticks = min(np.ceil(casting_frames/phantasm_tick_rate), 10)
    return num_ticks / casting_frames


def compute_channeling_playstyle(
        frames_per_animation: int,
        phantasm_tick_rate: int = 60,
        num_phantasms: int = 3) -> Tuple[int, int]:
    """Computes number of ticks for one phantasm in a channeling playstyle cycle.

    Args:
        frames_per_animation: Spirit Barrage breakpoint
        phantasm_tick_rate: damage instance tick rate of phantasms
        num_phantasms: maximum number of phantasms

    Returns:
        Tuple[int, int]: number of ticks occurred during cycle, frames needed to resolve animations in cycle
    """
    casting_frames = frames_per_animation * num_phantasms
    num_ticks = min(np.ceil(casting_frames/phantasm_tick_rate), 10)
    return num_ticks, casting_frames


def compute_delaying_playstyle(
        frames_per_animation: int,
        phantasm_tick_rate: int = 60,
        num_phantasms: int = 3) -> Tuple[int, int, int]:
    """Computes number of ticks for one phantasm in a delaying playstyle cycle.

    Args:
        frames_per_animation: Spirit Barrage breakpoint
        phantasm_tick_rate: damage instance tick rate of phantasms
        num_phantasms: maximum number of phantasms

    Returns:
        Tuple[int, int, int]: number of ticks during cycle, frames needed to resolve animations in cycle, frames to
            delay animations with in cycle
    """
    casting_frames = frames_per_animation * num_phantasms
    num_ticks = min(np.ceil(casting_frames/phantasm_tick_rate), 9)
    next_tick_frame = num_ticks * phantasm_tick_rate + 1
    delaying_frames = next_tick_frame - casting_frames - 1
    return num_ticks + 1, casting_frames, delaying_frames + 3


def compute_sniping_playstyle(
        frames_per_animation: int,
        engine_tick_rate: int = 60,
        phantasm_tick_rate: int = 60,
        phantasm_life_time: int = 10,
        num_phantasms: int = 3) -> Tuple[int, int, int]:
    """Computes number of ticks for one phantasm in a sniping playstyle cycle.

    Args:
        frames_per_animation: Spirit Barrage breakpoint
        engine_tick_rate: server tick rate
        phantasm_tick_rate: damage instance tick rate of phantasms
        phantasm_life_time: number of seconds a phantasm is alive
        num_phantasms: maximum number of phantasms

    Returns:
        Tuple[int, int, int]: number of ticks during cycle, frames needed to resolve animations in cycle, frames to
            snipe animation with in cycle
    """
    casting_frames = frames_per_animation * num_phantasms
    all_frame_snipes = list(range(0, engine_tick_rate * phantasm_life_time - 3 * frames_per_animation))
    all_frame_dps = [
        (
            i,
            compute_phantasm_ticks_per_frame(
                frames_per_animation=frames_per_animation,
                casting_delay_frames=i,
                phantasm_tick_rate=phantasm_tick_rate
            )
         ) for i in all_frame_snipes
    ]
    all_frame_dps = sorted(all_frame_dps, key=lambda x: (-x[1], -x[0]))
    snipe_frame = all_frame_dps[0][0]
    num_ticks = min(np.ceil((casting_frames+snipe_frame)/phantasm_tick_rate), 10)
    return num_ticks, casting_frames, snipe_frame


def compute_full_duration_playstyle(
        engine_tick_rate: int,
        phantasm_life_time: int,
        phantasm_tick_rate: int) -> Tuple[int, int]:
    """Computer number of ticks for one phantasm in a full-duration playstyle cycle.

    Args:
        engine_tick_rate: server tick rate
        phantasm_life_time: number of seconds a phantasm is alive
        phantasm_tick_rate: damage instance tick rate of phantasms

    Returns:
        Tuple[int, int]: number of ticks during cycle, frames needed to resolve playstyle per cycle
    """
    num_ticks = (phantasm_life_time * engine_tick_rate) // phantasm_tick_rate
    return num_ticks, phantasm_life_time * engine_tick_rate


def plot(
        x: np.ndarray,
        ys: Sequence[np.ndarray],
        labels: Sequence[str],
        lines: Sequence[str],
        markers: Optional[Sequence] = None,
        xlabel: str = 'x',
        ylabel: str = 'y',
        title: str = '',
        filename: str = 'plot.png',
        vlines: Optional[np.ndarray] = None,
        grid: bool = False,
        xticks: Optional[np.ndarray] = None,
        yticks: Optional[np.ndarray] = None,
        alpha: float = 1.0) -> None:
    """Plots n-many lines in ys over x.

    Args:
        x: values for x-axis
        ys: values for y-axis for n-line plots
        labels: label for each line
        lines: linetype settings for each line
        markers: marker for each line
        xlabel: x-axis label
        ylabel: y-axis label
        title: title of plot
        filename: filename output
        vlines: adds vertical lines to plot
        grid: activates grid
        xticks: sets custom xticks
        yticks: sets custom yticks
        alpha: transparency value for lines
    """
    # get ticks and ranges for input data
    xticks_start, xticks_stop = np.floor(np.min(x)), np.ceil(np.max(x))
    yticks_start, yticks_stop = np.floor(np.min(ys)), np.ceil(np.max(ys))
    xticks = np.arange(xticks_start, xticks_stop, 1) if xticks is None else xticks
    yticks = np.arange(yticks_start, yticks_stop, 10) if yticks is None else yticks
    xrange = (xticks[0], xticks[-1])
    yrange = (yticks[0], yticks[-1])

    fig, ax = plt.subplots()
    if markers is not None:
        for y, line, label, marker in zip(ys, lines, labels, markers):
            ax.plot(x, y, line, label=label, linewidth=0.75, marker=marker, alpha=alpha)
    else:
        for y, line, label in zip(ys, lines, labels):
            ax.plot(x, y, line, label=label, linewidth=0.75, alpha=alpha)
    if vlines is not None:
        ax.vlines(vlines, ymin=yrange[0], ymax=yrange[-1], linestyles='dotted', linewidth=0.75)

    # set ticks and ranges
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    # label, grid, legend, export
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if grid:
        if vlines is not None:
            ax.yaxis.grid()
        else:
            ax.grid()
    ax.legend()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(filename, dpi=300)


def step(
        x: np.ndarray,
        ys: Sequence[np.ndarray],
        labels: Sequence[str],
        lines: Sequence[str],
        xlabel: str = 'x',
        ylabel: str = 'y',
        title: str = '',
        filename: str = 'plot.png',
        vlines: Optional[np.ndarray] = None,
        grid: bool = False,
        xticks: Optional[np.ndarray] = None,
        yticks: Optional[np.ndarray] = None,
        alpha: float = 1.0) -> None:
    """Plots n-many step plots in ys over x.
    Args:
        x: values for x-axis
        ys: values for y-axis for n-line plots
        labels: label for each line
        lines: linetype settings for each line
        xlabel: x-axis label
        ylabel: y-axis label
        title: title of plot
        filename: filename output
        vlines: adds vertical lines to plot
        grid: activates grid
        xticks: sets custom xticks
        yticks: sets custom yticks
        alpha: transparency value for lines
    """
    # get ticks and ranges for input data
    xticks_start, xticks_stop = np.floor(np.min(x)), np.ceil(np.max(x))
    yticks_start, yticks_stop = np.floor(np.min(ys)), np.ceil(np.max(ys))
    xticks = np.arange(xticks_start, xticks_stop, 1) if xticks is None else xticks
    yticks = np.arange(yticks_start, yticks_stop, 10) if yticks is None else yticks
    xrange = (xticks[0], xticks[-1])
    yrange = (yticks[0], yticks[-1])

    fig, ax = plt.subplots()
    for y, line, label in zip(ys, lines, labels):
        ax.step(x, y, line, label=label, alpha=alpha)
    if vlines is not None:
        ax.vlines(vlines, ymin=yrange[0], ymax=yrange[-1], linestyles='dotted', linewidth=0.75)

    # set ticks and ranges
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    # label, grid, legend, export
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    if grid:
        if vlines is not None:
            ax.yaxis.grid()
        else:
            ax.grid()
    ax.legend()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(filename, dpi=300)


def main(
        patch_version: str,
        engine_tick_rate: int = 60,
        speed_coefficient: float = 1.2,
        base_animation_length: int = 21,
        first_breakpoint: int = 44,
        last_breakpoint: int = 9,
        phantasm_tick_rate: int = 30,
        phantasm_life_time: int = 10,
        phantasm_tick_multiplier: float = 1.0,
        num_phantasms: int = 3,
        attack_speed_delta: float = 0.001) -> None:
    """Computes everything we need to know about Spirit Barrage with Phantasm.

    Args:
        patch_version: title in all plots and prefix for file outputs
        engine_tick_rate: server tick rate
        speed_coefficient: separate attack speed multiplier for spirit barrage
        base_animation_length: animation length of spirit barrage
        first_breakpoint: starting breakpoint to calculate from (low aps)
        last_breakpoint: last breakpoint to calculate for (high aps)
        phantasm_tick_rate: application rate of damage instances of phantasms in frames
        phantasm_life_time: duration of phantasm rune in seconds
        phantasm_tick_multiplier: per tick hit multiplier of phantasms
        num_phantasms: max number of phantasms alive at the same time
        attack_speed_delta: step value for attack speed plots

    Returns:
        Magic and Knowledge.
    """
    assert first_breakpoint > last_breakpoint
    assert first_breakpoint > 0
    assert speed_coefficient > 0
    assert base_animation_length > 0
    assert phantasm_tick_rate > 0
    assert phantasm_life_time > 0
    assert phantasm_tick_multiplier > 0
    assert engine_tick_rate > 0
    assert num_phantasms > 0
    assert attack_speed_delta > 0
    # calculate Spirit Barrage Breakpoint table
    breakpoints = np.asarray(range(first_breakpoint, last_breakpoint - 1, -1), dtype=np.int32)
    spirit_barrage_breakpoint_table = [
        (i,
         compute_character_attack_speed(
                frames_per_animation=i,
                base_animation_length=base_animation_length,
                speed_coefficient=speed_coefficient,
                engine_tick_rate=engine_tick_rate
            )
         ) for i in breakpoints
    ]

    # plot DPS for channeling-playstyle and its breakpoint anomaly
    phantasm_channel_ticks, channel_frames = zip(*[
        compute_channeling_playstyle(
            frames_per_animation=i,
            phantasm_tick_rate=phantasm_tick_rate,
            num_phantasms=num_phantasms
        ) for i in breakpoints]
    )
    phantasm_channel_ticks = np.asarray(phantasm_channel_ticks, dtype=np.int32)
    channel_frames = np.asarray(channel_frames, dtype=np.int32)
    channel_duration = channel_frames
    channel_tps = num_phantasms * phantasm_channel_ticks * engine_tick_rate / channel_duration

    # calculate delaying-playstyle frames
    phantasm_delay_ticks, casting_frames, delay_frames = zip(*[
        compute_delaying_playstyle(
            frames_per_animation=i,
            phantasm_tick_rate=phantasm_tick_rate,
            num_phantasms=num_phantasms
        ) for i in breakpoints]
    )
    phantasm_delay_ticks = np.asarray(phantasm_delay_ticks, dtype=np.int32)
    delay_frames = np.asarray(delay_frames, dtype=np.int32)
    casting_frames = np.asarray(casting_frames, dtype=np.int32)
    delay_duration = delay_frames + casting_frames
    delay_tps = num_phantasms * phantasm_delay_ticks * engine_tick_rate / delay_duration

    # calculate full-duration DPS
    phantasm_duration_ticks, full_frames = zip(*[
        compute_full_duration_playstyle(
            engine_tick_rate=engine_tick_rate,
            phantasm_life_time=phantasm_life_time,
            phantasm_tick_rate=phantasm_tick_rate
        ) for _ in breakpoints]
    )
    phantasm_duration_ticks = np.asarray(phantasm_duration_ticks, dtype=np.int32)
    full_frames = np.asarray(full_frames, dtype=np.int32)
    full_duration = full_frames
    full_tps = num_phantasms * phantasm_duration_ticks * engine_tick_rate / full_duration

    # calculate snipe-duration frames
    phantasm_snipe_ticks, casting_frames, snipe_frames = zip(*[
        compute_sniping_playstyle(
            frames_per_animation=i,
            engine_tick_rate=60,
            phantasm_tick_rate=phantasm_tick_rate,
            phantasm_life_time=phantasm_life_time,
            num_phantasms=num_phantasms
        ) for i in breakpoints]
    )
    phantasm_snipe_ticks = np.asarray(phantasm_snipe_ticks, dtype=np.int32)
    snipe_frames = np.asarray(snipe_frames, dtype=np.int32)
    casting_frames = np.asarray(casting_frames, dtype=np.int32)
    snipe_duration = casting_frames + snipe_frames
    snipe_tps = num_phantasms * phantasm_snipe_ticks * engine_tick_rate / snipe_duration

    # calculate character_speed_values for x-axis
    start_attack_speed = compute_character_attack_speed(
        frames_per_animation=first_breakpoint,
        base_animation_length=base_animation_length,
        speed_coefficient=speed_coefficient,
        engine_tick_rate=engine_tick_rate
    )[0]
    stop_attack_speed = compute_character_attack_speed(
        frames_per_animation=last_breakpoint,
        base_animation_length=base_animation_length,
        speed_coefficient=speed_coefficient,
        engine_tick_rate=engine_tick_rate
    )[1]
    attacks_per_second_values = np.arange(start_attack_speed, stop_attack_speed, attack_speed_delta)
    breakpoints_over_aps = np.asarray([
        compute_frames_per_animation(
            attacks_per_second=i,
            base_animation_length=base_animation_length,
            speed_coefficient=speed_coefficient,
            engine_tick_rate=engine_tick_rate,
            is_channeling=False
        ) for i in attacks_per_second_values
    ], dtype=np.float64)

    # calculate dps: multiply playstyle tps with aps and phantasm_tick_multiplier
    channel_dps, delay_dps, snipe_dps, full_dps = [], [], [], []
    for aps, bp in zip(attacks_per_second_values, breakpoints_over_aps):
        index = np.argwhere(breakpoints == bp)[0][0]
        channel_dps.append(aps * phantasm_tick_multiplier * channel_tps[index])
        delay_dps.append(aps * phantasm_tick_multiplier * delay_tps[index])
        snipe_dps.append(aps * phantasm_tick_multiplier * snipe_tps[index])
        full_dps.append(aps * phantasm_tick_multiplier * full_tps[index])
    channel_dps = np.asarray(channel_dps, dtype=np.float64)
    delay_dps = np.asarray(delay_dps, dtype=np.float64)
    snipe_dps = np.asarray(snipe_dps, dtype=np.float64)
    full_dps = np.asarray(full_dps, dtype=np.float64)

    # file patch prefix
    file_prefix = patch_version.lower().replace(' ', '').replace('.', '')

    # plot channeling-, delaying-, snipe-, full-duration DPS
    ys = [phantasm_channel_ticks, phantasm_snipe_ticks, phantasm_delay_ticks, phantasm_duration_ticks]
    plot(
        x=breakpoints,
        ys=ys,
        labels=['Channeling', 'Sniping', 'Delaying', 'Full'],
        lines=['b', 'g', 'r', 'c'],
        markers=[4, 5, 6, 7],
        xlabel='Frames per Animation',
        ylabel='Phantasm Ticks',
        title=f'Ticks for each Phantasm over FPA in {patch_version}',
        filename=f'{file_prefix}_ticks.png',
        grid=True,
        xticks=np.arange(last_breakpoint, first_breakpoint + 1, 1),
        yticks=np.arange(0, np.max(ys) + 2, 1),
        alpha=0.75,
    )
    ys = [channel_duration, snipe_duration, delay_duration, full_duration]
    plot(
        x=breakpoints,
        ys=ys,
        labels=['Channeling', 'Sniping', 'Delaying', 'Full'],
        lines=['b', 'g', 'r', 'c'],
        markers=[4, 5, 6, 7],
        xlabel='Frames per Animation',
        ylabel='Duration in Frames',
        title=f'Duration over FPA until oldest Phantasm detonates in {patch_version}',
        xticks=np.arange(last_breakpoint, first_breakpoint + 1, 1),
        yticks=np.arange(0, np.max(ys) + 50, 25),
        grid=True,
        alpha=0.75,
        filename=f'{file_prefix}_duration.png',
    )
    ys = [channel_tps, snipe_tps, delay_tps, full_tps]
    plot(
        x=breakpoints,
        ys=ys,
        labels=['Channeling', 'Sniping', 'Delaying', 'Full'],
        lines=['b', 'g', 'r', 'c'],
        markers=[4, 5, 6, 7],
        xlabel='Frames per Animation',
        ylabel='Phantasm Ticks per Second',
        title=f'Phantasm Ticks per Second over FPA in {patch_version}',
        xticks=np.arange(last_breakpoint, first_breakpoint + 1, 1),
        yticks=np.arange(np.min(ys) - 1, np.max(ys) + 2, 1),
        grid=True,
        alpha=0.75,
        filename=f'{file_prefix}_tps.png',
    )
    ys = [snipe_frames, delay_frames/num_phantasms]
    plot(
        x=breakpoints,
        ys=ys,
        labels=['Sniping', 'Delaying'],
        lines=['g', 'b'],
        markers=['o', 'o'],
        xlabel='Frames per Animation',
        ylabel='Optimal Frames',
        title=f'Optimal Frames over FPA in {patch_version}',
        xticks=np.arange(last_breakpoint, first_breakpoint + 1, 1),
        yticks=np.arange(0, np.max(ys) + 2, 1),
        grid=True,
        alpha=0.75,
        filename=f'{file_prefix}_frames.png',
    )
    x = attacks_per_second_values
    ys = [breakpoints_over_aps]
    step(
        x=attacks_per_second_values,
        ys=ys,
        labels=['Spirit Barrage'],
        lines=['b'],
        xlabel='Attacks Per Second',
        ylabel='Frames Per Animation',
        title=f'Breakpoints for Spirit Barrage (speed_coefficient={speed_coefficient}, '
              f'base_animation_length={base_animation_length}) over APS in Patch {patch_version}',
        vlines=np.asarray([bp[1][0] for bp in spirit_barrage_breakpoint_table], dtype=np.float64),
        xticks=np.arange(np.floor(x[0]), np.ceil(x[-1]) + 1, 1),
        yticks=np.arange(np.min(ys) - 1, np.max(ys) + 2, 1),
        grid=True,
        filename=f'{file_prefix}_bp.png',
    )
    x = attacks_per_second_values
    ys = [channel_dps, snipe_dps, delay_dps, full_dps]
    plot(
        x=x,
        ys=ys,
        labels=['Channeling', 'Sniping', 'Delaying', 'Full'],
        lines=['b', 'g', 'r', 'c'],
        xlabel='Attacks Per Second',
        ylabel='Phantasm Multiplier',
        title=f'Phantasm Multiplier for Spirit Barrage over APS in Patch {patch_version}',
        vlines=np.asarray([bp[1][0] for bp in spirit_barrage_breakpoint_table], dtype=np.float64),
        xticks=np.arange(np.floor(x[0]), np.ceil(x[-1]) + 1, 1),
        yticks=np.arange(0, np.max(ys) + 20, 10),
        grid=True,
        alpha=0.75,
        filename=f'{file_prefix}_dps.png',
    )


if __name__ == '__main__':
    main(patch_version='Patch 2.6.8', phantasm_tick_rate=30, phantasm_life_time=10, phantasm_tick_multiplier=1.0)
