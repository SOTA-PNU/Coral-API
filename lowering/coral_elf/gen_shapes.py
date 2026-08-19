#!/usr/bin/env python3
"""manifest.json → model_shapes.h  (launcher 가 쓰는 모델별 상수)

EmitC 경로에는 VMFB 가 없어 iree-dump-module 로 ABI 를 읽을 수 없다.
그래서 ① 단계가 남긴 manifest 가 유일한 ABI 출처가 된다.

생성하는 것:
    MODEL_FUNCTION_NAME     "module.main"
    MODEL_LIBRARY_QUERY     <name>_library_query   (model.h 에서 파싱)
    MODEL_INPUT_COUNT / MODEL_OUTPUT_COUNT
    model_inputs[]          입력 데이터 + shape + dtype 테이블
    model_output_elems[]    출력별 원소 수
    MODEL_OUTPUT_TOTAL_ELEMS

배치
----
ITCM 은 1 MiB 뿐이라 큰 배열은 .rodata(=ITCM) 에 두면 링크가 안 된다.
DDR_THRESHOLD 를 넘는 배열은 링커 스크립트의 .ddr_data 섹션으로 보낸다.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

ETYPE = {
    "f32": ("IREE_HAL_ELEMENT_TYPE_FLOAT_32", "float", 4, "f"),
    "f16": ("IREE_HAL_ELEMENT_TYPE_FLOAT_16", "uint16_t", 2, "H"),
    "i8":  ("IREE_HAL_ELEMENT_TYPE_INT_8", "int8_t", 1, "b"),
    "i32": ("IREE_HAL_ELEMENT_TYPE_INT_32", "int32_t", 4, "i"),
    "i64": ("IREE_HAL_ELEMENT_TYPE_INT_64", "int64_t", 8, "q"),
}

# 이 크기를 넘는 배열은 DDR 로 보낸다 (ITCM 1 MiB 보호).
DDR_THRESHOLD = 64 * 1024

QUERY_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*_library_query)\s*\(", re.M)


def _cfloat(v):
    """C float 리터럴. '0f' 는 8진수로 파싱되므로 소수점을 반드시 넣는다."""
    s = f"{v:.9g}"
    if not any(ch in s for ch in ".eEnN"):     # nan/inf 는 그대로 둔다
        s += ".0"
    return s + "f"


def _emit_array(lines, ctype, name, values, dtype, section):
    parts = ["aligned(64)"]
    if section:
        parts.append(f'section("{section}")')
    attrs = "__attribute__((" + ", ".join(parts) + "))"
    lines.append(f"static const {ctype} {name}[{len(values)}] {attrs} = {{")
    per_line = 12 if dtype in ("f32", "f16") else 20
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        if dtype == "f32":
            body = ", ".join(_cfloat(v) for v in chunk)
        else:
            body = ", ".join(str(v) for v in chunk)
        lines.append("    " + body + ",")
    lines.append("};")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", type=Path, required=True,
                    help="export_tosa.py 가 만든 build/<model>/")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.build_dir / "manifest.json").read_text())
    abi = manifest["abi"]
    inputs, outputs = abi["inputs"], abi["outputs"]

    # query 심볼은 iree-compile 이 만든 model.h 에서 읽는다 (모델마다 다름).
    header = (args.build_dir / "model.h").read_text(encoding="utf-8")
    queries = QUERY_RE.findall(header)
    if len(queries) != 1:
        print(f"error: model.h 에서 *_library_query 를 1개 찾지 못했습니다: {queries}",
              file=sys.stderr)
        return 1
    query = queries[0]

    lines = [
        "// === 자동 생성. 수정하지 마세요. ===",
        "// gen_shapes.py 가 manifest.json + model.h 로부터 만들었습니다.",
        f"// model: {manifest['model']}",
        "#ifndef MODEL_SHAPES_H_",
        "#define MODEL_SHAPES_H_",
        "",
        '#include "iree/hal/api.h"',
        "",
        f'#define MODEL_FUNCTION_NAME "module.{abi["function"]}"',
        f"#define MODEL_LIBRARY_QUERY {query}",
        f"#define MODEL_INPUT_COUNT   {len(inputs)}",
        f"#define MODEL_OUTPUT_COUNT  {len(outputs)}",
        "",
        "typedef struct {",
        "  const void* data;",
        "  iree_host_size_t bytes;",
        "  const iree_hal_dim_t* shape;",
        "  iree_host_size_t rank;",
        "  iree_hal_element_type_t etype;",
        "} model_input_t;",
        "",
    ]

    ddr_used = 0
    for src in inputs:
        i = src["index"]
        dtype = src["dtype"]
        if dtype not in ETYPE:
            print(f"error: 지원하지 않는 dtype {dtype}", file=sys.stderr)
            return 1
        enum, ctype, esize, fmt = ETYPE[dtype]

        blob = (args.build_dir / src["file"]).read_bytes()
        if len(blob) != src["bytes"]:
            print(f"error: {src['file']} 크기 불일치 "
                  f"({len(blob)} != {src['bytes']})", file=sys.stderr)
            return 1
        values = struct.unpack(f"<{len(blob)//esize}{fmt}", blob)

        section = ".ddr_data" if len(blob) > DDR_THRESHOLD else None
        if section:
            ddr_used += len(blob)
        lines.append(f"// input{i}: shape {src['shape']} {dtype} "
                     f"({len(blob)} bytes{', DDR' if section else ''})")
        _emit_array(lines, ctype, f"model_input{i}_data", values, dtype, section)
        shape = ", ".join(str(d) for d in src["shape"])
        lines.append(f"static const iree_hal_dim_t model_input{i}_shape[] = "
                     f"{{{shape}}};")
        lines.append("")

    lines.append(f"static const model_input_t model_inputs[MODEL_INPUT_COUNT] = {{")
    for src in inputs:
        i = src["index"]
        enum = ETYPE[src["dtype"]][0]
        lines.append(f"    {{model_input{i}_data, sizeof(model_input{i}_data), "
                     f"model_input{i}_shape, {len(src['shape'])}, {enum}}},")
    lines.append("};")
    lines.append("")

    total = 0
    for o in outputs:
        n = o["bytes"] // ETYPE[o["dtype"]][2]
        lines.append(f"#define MODEL_OUTPUT{o['index']}_ELEMS {n}"
                     f"   // shape {o['shape']} {o['dtype']}")
        total += n
    lines.append(f"#define MODEL_OUTPUT_TOTAL_ELEMS {total}")
    lines.append("static const iree_host_size_t "
                 "model_output_elems[MODEL_OUTPUT_COUNT] = {"
                 + ", ".join(str(o["bytes"] // ETYPE[o["dtype"]][2])
                             for o in outputs) + "};")
    # 출력 버퍼도 크면 DDR 로 (BSS 쪽).
    out_bytes = total * 4
    lines.append("")
    if out_bytes > DDR_THRESHOLD:
        lines.append('#define MODEL_OUTPUT_SECTION __attribute__((section(".ddr_bss")))')
    else:
        lines.append("#define MODEL_OUTPUT_SECTION")
    lines.append("")
    lines.append("#endif  // MODEL_SHAPES_H_")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[gen_shapes] function : module.{abi['function']}")
    print(f"[gen_shapes] query    : {query}")
    for src in inputs:
        print(f"[gen_shapes] input{src['index']}   : {src['shape']} {src['dtype']} "
              f"({src['bytes']} bytes)")
    for o in outputs:
        print(f"[gen_shapes] output{o['index']}  : {o['shape']} {o['dtype']}")
    print(f"[gen_shapes] DDR 배치 : {ddr_used/1024:.0f} KB 입력, "
          f"출력 {out_bytes/1024:.0f} KB "
          f"({'DDR' if out_bytes > DDR_THRESHOLD else 'DTCM'})")
    print(f"[gen_shapes] wrote    : {args.output} "
          f"({args.output.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
