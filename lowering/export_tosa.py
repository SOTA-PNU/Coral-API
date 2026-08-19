#!/usr/bin/env python3
"""① PyTorch 모델 → ② TOSA MLIR.

모델과 무관한 공통 스크립트다. 모델별 정보는 전부 models/model_<name>.py 의
컨트랙트 함수에서 온다:

    get_model()               -> nn.Module (eval)
    get_example_inputs()      -> tuple[Tensor, ...]
    get_calibration_batches() -> Iterable[tuple[Tensor, ...]]   (선택)
    get_calibration_info()    -> dict                            (선택)

사용:
    python export_tosa.py lenet5
    python export_tosa.py vgg_backbone --outdir build
    python export_tosa.py charrnn --no-quantize      # float 경로만 확인

산출물 (build/<name>/):
    model.mlirbc      TOSA (bytecode). iree-compile 이 그대로 받는다.
    manifest.json     ABI·양자화·데이터 출처·op 통계. 이후 단계가 전부 이걸 읽는다.
    input<i>.bin      example input (raw). launcher/검증용.
    ref_output<i>.bin 기준 출력 (raw). 회귀 게이트.

각 단계에 게이트가 있다. 실패하면 그 자리에서 멈추고 이유를 말한다.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import re
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# torch dtype -> (짧은 이름, 바이트/원소)
DTYPE_INFO = {
    torch.float32: ("f32", 4), torch.float16: ("f16", 2),
    torch.float64: ("f64", 8), torch.int64: ("i64", 8),
    torch.int32: ("i32", 4), torch.int16: ("i16", 2),
    torch.int8: ("i8", 1), torch.uint8: ("u8", 1), torch.bool: ("i1", 1),
}


class Stage:
    """단계별 진행 표시. 실패하면 예외를 그대로 올린다."""

    def __init__(self, total):
        self.total = total
        self.n = 0

    def __call__(self, label):
        self.n += 1
        print(f"[{self.n}/{self.total}] {label:.<24} ", end="", flush=True)

    @staticmethod
    def ok(detail=""):
        print(f"OK   {detail}")

    @staticmethod
    def skip(detail=""):
        print(f"SKIP {detail}")


def describe(tensors, files=None):
    """텐서 목록 -> manifest 용 dict 목록."""
    out = []
    for i, t in enumerate(tensors):
        name, esize = DTYPE_INFO.get(t.dtype, (str(t.dtype), t.element_size()))
        entry = {"index": i, "shape": list(t.shape), "dtype": name,
                 "bytes": t.numel() * esize}
        if files:
            entry["file"] = files[i]
        out.append(entry)
    return out


def as_tuple(x):
    return x if isinstance(x, (tuple, list)) else (x,)


def promote_frozen_params(model):
    """PT2E 가 buffer 로 등록한 int8 가중치를 Parameter 로 재분류한다.

    torch-mlir 의 frozen importer 는 parameter 와 lifted constant 만 상수로
    접고 persistent buffer 는 접지 않는다. 승격하지 않으면 가중치가 전부
    함수 인자로 새어 나와 ABI 가 오염된다.
    """
    promoted = []
    for name, value in list(model.named_buffers()):
        if not name.startswith("_frozen_param"):
            continue
        delattr(model, name)
        model.register_parameter(
            name, torch.nn.Parameter(value.detach(), requires_grad=False))
        promoted.append(name)
    leftover = [n for n, _ in model.named_buffers()
                if n.startswith("_frozen_param")]
    assert not leftover, f"승격 후에도 남은 buffer: {leftover}"
    return promoted


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="models/model_<name>.py 의 <name>")
    ap.add_argument("--outdir", default="build", help="산출물 상위 디렉터리")
    ap.add_argument("--no-quantize", action="store_true",
                    help="PT2E 를 건너뛰고 float TOSA 만 생성")
    ap.add_argument("--per-channel", action="store_true",
                    help="per-channel 양자화 (정수 rescale 경로가 죽을 수 있음)")
    ap.add_argument("--text", action="store_true",
                    help="텍스트 MLIR 도 저장 (큰 모델에서는 수백 MB)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    outdir = Path(args.outdir) / args.model
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    stage = Stage(6)

    # ---- ① 모델 플러그인 로드 ------------------------------------------
    stage("모델 로드")
    mod = importlib.import_module(f"models.model_{args.model}")
    for fn in ("get_model", "get_example_inputs"):
        if not hasattr(mod, fn):
            raise SystemExit(f"\n  error: models/model_{args.model}.py 에 "
                             f"{fn}() 가 없습니다 (컨트랙트 위반)")
    model = mod.get_model()
    example = as_tuple(mod.get_example_inputs())
    n_param = sum(p.numel() for p in model.parameters())
    stage.ok(f"{args.model}  {n_param/1e6:.2f} M param  "
             f"입력 {len(example)}개")

    # ---- ② export 게이트 (PT2E 전에!) -----------------------------------
    stage("export 게이트")
    ep = torch.export.export(model, example)
    stage.ok(f"{len(list(ep.graph.nodes))} nodes")

    with torch.no_grad():
        float_out = as_tuple(model(*example))

    # ---- ③ PT2E 양자화 ---------------------------------------------------
    calib_info, promoted = {}, []
    do_quantize = not args.no_quantize and hasattr(mod, "get_calibration_batches")

    if do_quantize:
        # 이 import 들이 stderr 로 경고를 뱉으므로 진행 표시 전에 끝내둔다.
        from torch.export import export_for_training
        from torchao.quantization.pt2e.quantize_pt2e import (
            convert_pt2e, prepare_pt2e)
        try:
            from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
        except ImportError:
            from torch.ao.quantization.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer, get_symmetric_quantization_config)
    from torch_mlir import fx          # 여기도 첫 import 시 로그가 나온다

    stage("양자화")
    if not do_quantize:
        stage.skip("float 경로")
        qmodel = model
    else:
        captured = export_for_training(model, example).module()
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(
                is_per_channel=args.per_channel, is_dynamic=False))
        prepared = prepare_pt2e(captured, quantizer)
        n_batch = 0
        with torch.no_grad():
            for batch in mod.get_calibration_batches():
                prepared(*as_tuple(batch))
                n_batch += 1
        if n_batch == 0:
            raise SystemExit("\n  error: calibration 배치가 0개입니다")
        qmodel = convert_pt2e(prepared)
        promoted = promote_frozen_params(qmodel)
        if not promoted:
            raise SystemExit(
                "\n  error: _frozen_param buffer 가 하나도 없습니다.\n"
                "         PT2E 네이밍이 바뀌었거나 양자화 대상 layer 가 없습니다.")
        calib_info = (mod.get_calibration_info()
                      if hasattr(mod, "get_calibration_info") else {})
        stage.ok(f"{n_batch} batch, {len(promoted)}개 승격"
                 + ("  [per-channel]" if args.per_channel else ""))

    with torch.no_grad():
        quant_out = as_tuple(qmodel(*example))

    # ---- ④ TOSA ---------------------------------------------------------
    stage("TOSA 변환")
    module = fx.export_and_import(qmodel, *example, output_type="tosa",
                                  func_name="main")
    asm = module.operation.get_asm()
    signature = next(l.strip() for l in asm.splitlines()
                     if "func.func @main" in l)
    stage.ok(f"{len(asm)/1e6:.1f} MB")

    # ---- ⑤ 검증 게이트 ---------------------------------------------------
    stage("검증")
    n_args = signature.count("%arg")
    if n_args != len(example):
        raise SystemExit(
            f"\n  error: ABI 누수 — 인자 {n_args}개 (기대 {len(example)}개)\n"
            f"         가중치가 함수 인자로 새어 나왔습니다.\n"
            f"         {signature[:180]}")
    ops = collections.Counter(re.findall(r"tosa\.[a-z_0-9]+", asm))
    integer_ops = ops["tosa.conv2d"] + ops["tosa.matmul"]
    stage.ok(f"입력 {n_args} / 출력 {len(quant_out)}   "
             f"conv2d={ops['tosa.conv2d']} matmul={ops['tosa.matmul']} "
             f"rescale={ops['tosa.rescale']} floor={ops['tosa.floor']}")

    # ---- ⑥ 저장 ----------------------------------------------------------
    stage("저장")
    with open(outdir / "model.mlirbc", "wb") as f:
        module.operation.write_bytecode(f)
    if args.text:
        (outdir / "model.mlir").write_text(asm, encoding="utf-8")

    in_files, out_files = [], []
    for i, t in enumerate(example):
        name = f"input{i}.bin"
        t.detach().contiguous().numpy().tofile(outdir / name)
        in_files.append(name)
    for i, t in enumerate(quant_out):
        name = f"ref_output{i}.bin"
        t.detach().contiguous().numpy().tofile(outdir / name)
        out_files.append(name)

    # 양자화 오차: float 기준 대비 얼마나 벌어졌나
    quant_error = None
    if len(float_out) == len(quant_out):
        try:
            quant_error = max(float((a - b).abs().max())
                              for a, b in zip(float_out, quant_out))
        except Exception:
            quant_error = None

    manifest = {
        "model": args.model,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "abi": {
            "function": "main",
            "signature": signature,
            "inputs": describe(example, in_files),
            "outputs": describe(quant_out, out_files),
        },
        "params": {"count": n_param, "int8_bytes": n_param},
        "quantization": {
            "enabled": not args.no_quantize and bool(promoted),
            "scheme": ("pt2e-xnnpack-symmetric-per-channel" if args.per_channel
                       else "pt2e-xnnpack-symmetric-per-tensor"),
            "frozen_params": promoted,
            "max_abs_error_vs_float": quant_error,
        },
        "calibration": calib_info,
        "tosa_ops": dict(ops.most_common()),
        "integer_core_ops": integer_ops,
        "artifacts": {"mlirbc": "model.mlirbc",
                      "text_mlir": "model.mlir" if args.text else None},
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    stage.ok(str(outdir))

    print()
    print(f"  ABI      {signature[:150]}")
    if quant_error is not None:
        print(f"  양자화오차 {quant_error:.6g}  (float 기준 대비)")
    if calib_info and not calib_info.get("real_data", True):
        print(f"  ⚠  calibration 이 실데이터가 아닙니다 — 정확도 평가 불가")
    print(f"  소요     {manifest['elapsed_sec']}s")


if __name__ == "__main__":
    main()
