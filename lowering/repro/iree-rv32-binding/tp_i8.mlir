func.func @main(%arg0: tensor<64x56x56xi8>) -> tensor<58x58x64xi8> {
  %c0_i8 = arith.constant 0 : i8
  %e1 = tensor.empty() : tensor<56x56x64xi8>
  %t = linalg.generic {indexing_maps = [affine_map<(a,b,c)->(c,a,b)>, affine_map<(a,b,c)->(a,b,c)>],
        iterator_types = ["parallel","parallel","parallel"]}
        ins(%arg0 : tensor<64x56x56xi8>) outs(%e1 : tensor<56x56x64xi8>) {
    ^bb0(%x: i8, %o: i8):
      linalg.yield %x : i8
  } -> tensor<56x56x64xi8>
  %p = tensor.pad %t low[1, 1, 0] high[1, 1, 0] {
    ^bb0(%i0: index, %i1: index, %i2: index):
      tensor.yield %c0_i8 : i8
  } : tensor<56x56x64xi8> to tensor<58x58x64xi8>
  return %p : tensor<58x58x64xi8>
}
