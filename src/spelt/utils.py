def format_with_prefix(value: float) -> str:
    """Format a number with SI prefix.

    Args:
        value: Number to format

    Returns:
        Formatted string with SI prefix

    Examples:
        >>> format_with_prefix(0.000001)
        '1.00 µ'
        >>> format_with_prefix(1000)
        '1.00 k'
    """
    prefixes = {
        -24: 'y',  # yocto
        -21: 'z',  # zepto
        -18: 'a',  # atto
        -15: 'f',  # femto
        -12: 'p',  # pico
        -9: 'n',   # nano
        -6: 'µ',   # micro
        -3: 'm',   # milli
        0: '',     # unit
        3: 'k',    # kilo
        6: 'M',    # mega
        9: 'G',    # giga
        12: 'T',   # tera
        15: 'P',   # peta
        18: 'E',   # exa
        21: 'Z',   # zetta
        24: 'Y'    # yotta
    }

    if value == 0:
        return '0.00'

    import math
    exp = int(math.floor(math.log10(abs(value)) / 3) * 3)

    # Clamp to available prefixes
    exp = min(max(exp, min(prefixes.keys())), max(prefixes.keys()))

    scaled = value / 10**exp
    prefix = prefixes[exp]

    return f'{scaled:.2f} {prefix}'.strip()
