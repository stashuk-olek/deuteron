from __future__ import annotations
import re

_ATOM_F32: re.Pattern[str] = re.compile(
    r"^(\s*)(@%p\d+\s+)?atom((?:\.relaxed|\.release)?)((?:\.cta|\.cluster|\.gpu|\.sys)?)\.global\.add(\.noftz)?\.f32\s+(%r\d+),\s*(\[[^\]]+\]),\s*(%r\d+);\s*$"
)
_REGISTER_SUFFIX: re.Pattern[str] = re.compile(r"\d")
_MAXNTID: re.Pattern[str] = re.compile(r"\.maxntid\s+(\d+(?:\s*,\s*\d+\s*,\s*\d+)?)")
_DIV_RN_F32: re.Pattern[str] = re.compile(r"div\.rn\.f32(?=\s)")


def _register_used(rest: str, reg: str) -> bool:
    for match in re.finditer(re.escape(reg), rest):
        if not _REGISTER_SUFFIX.match(rest, match.end()):
            return True
    return False


def rewrite_dead_atom_to_red(ptx: str) -> str:
    lines: list[str] = ptx.split("\n")
    for index, line in enumerate(lines):
        match = _ATOM_F32.match(line)
        if match is None:
            continue
        indent, pred, sem, scope, noftz, dest, addr, src = match.groups()
        if _register_used("\n".join(lines[index + 1 :]), dest):
            continue
        lines[index] = (
            f"{indent}{pred or ''}red{sem}{scope}.global.add{noftz or ''}.f32 {addr}, {src};"
        )
    return "\n".join(lines)


def replace_maxntid_with_reqntid(ptx: str) -> str:
    return _MAXNTID.sub(r".reqntid \1", ptx)


def rewrite_div_rn_to_approx(ptx: str, approximate: bool) -> str:
    if not approximate:
        return ptx
    return _DIV_RN_F32.sub("div.approx.f32", ptx)


def rewrite_ptx(ptx: str, approximate: bool) -> str:
    return rewrite_div_rn_to_approx(
        replace_maxntid_with_reqntid(rewrite_dead_atom_to_red(ptx)), approximate
    )
