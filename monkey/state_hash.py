from __future__ import annotations

import hashlib
from typing import Iterable


def canonical_shots_tuple(shots: dict[int, int] | Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """
    将状态规范化为稳定的 (action_id, outcome) 排序元组。
    outcome: 0=MISS, 1=BODY, 2=HEAD
    """
    if isinstance(shots, dict):
        items = [(int(a), int(v)) for a, v in shots.items()]
    else:
        items = [(int(a), int(v)) for a, v in shots]
    items.sort(key=lambda x: x[0])
    return tuple(items)


def state_hash_hex(shots: dict[int, int] | Iterable[tuple[int, int]]) -> str:
    """
    对“坐标+状态”的列表做稳定哈希，用于 O(1) 查表。
    这里坐标用 action_id(0..99) 表示；保存到 JSON 时再转成 (x,y)。
    """
    t = canonical_shots_tuple(shots)
    # 100 格以内，每条 2 字节足够：a(0..99) + v(0..2)
    b = bytearray()
    for a, v in t:
        b.append(a & 0xFF)
        b.append(v & 0xFF)
    return hashlib.blake2s(bytes(b), digest_size=16).hexdigest()


