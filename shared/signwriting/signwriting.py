import re
from typing import List, Tuple, TypedDict


class SignSymbol(TypedDict):
    symbol: str
    position: Tuple[int, int]


class Sign(TypedDict):
    box: SignSymbol
    symbols: List[SignSymbol]


def fsw_to_sign(fsw: str) -> Sign:
    box = re.match(r'([BLMR])(\d{3})x(\d{3})', fsw)
    box_symbol, x, y = box.groups() if box is not None else ("M", 500, 500)

    symbols = re.findall(r'(S[123][0-9a-f]{2}[0-5][0-9a-f])(\d{3})x(\d{3})', fsw)

    return {
        "box": {
            "symbol": box_symbol,
            "position": (int(x), int(y))
        },
        "symbols": [{
            "symbol": s[0],
            "position": (int(s[1]), int(s[2]))
        } for s in symbols]
    }


def sign_to_fsw(sign: Sign) -> str:
    symbols = [sign["box"]] + sign["symbols"]
    symbols_str = [s["symbol"] + str(s["position"][0]) + 'x' + str(s["position"][1]) for s in symbols]
    return "".join(symbols_str)


def all_ys(_sign):
    return [s["position"][1] for s in _sign["symbols"]]


def join_signs(*fsws: str, spacing: int = 0):
    signs = [fsw_to_sign(fsw) for fsw in fsws]
    new_sign: Sign = {"box": {"symbol": "M", "position": (500, 500)}, "symbols": []}

    accumulative_offset = 0

    for sign in signs:
        sign_min_y = min(all_ys(sign))
        sign_offset_y = accumulative_offset + spacing - sign_min_y
        accumulative_offset += (sign["box"]["position"][1] - sign_min_y) + spacing  # * 2

        new_sign["symbols"] += [{
            "symbol": s["symbol"],
            "position": (s["position"][0], s["position"][1] + sign_offset_y)
        } for s in sign["symbols"]]

    # Recenter around box center
    sign_middle = max(all_ys(new_sign)) // 2

    for symbol in new_sign["symbols"]:
        symbol["position"] = (symbol["position"][0],
                              new_sign["box"]["position"][1] - sign_middle + symbol["position"][1])

    return sign_to_fsw(new_sign)
