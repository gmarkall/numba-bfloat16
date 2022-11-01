import numpy as np
import operator

from llvmlite import ir

from numba import cuda, types
from numba.core.types.scalars import Number
from numba.cuda.cudaimpl import registry
from numba.cuda.models import models, register_model
from numba.cuda.extending import intrinsic
from numba.core.extending import overload

lower_cast = registry.lower_cast


class BFloat16(Number):
    def __init__(self, *args, **kws):
        super().__init__(name='bfloat16')


bfloat16 = BFloat16()


@register_model(BFloat16)
class BFloat16Model(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(16))


@lower_cast(types.float32, bfloat16)
def float32_to_bfloat16(context, builder, fromty, toty, val):
    function_type = ir.FunctionType(ir.IntType(16), [ir.FloatType()])
    instruction = "cvt.rn.bf16.f32 $0, $1;"
    asm = ir.InlineAsm(function_type, instruction, "=h,h")
    return builder.call(asm, [val])


@lower_cast(bfloat16, types.float32)
def bfloat16_to_float32(context, builder, fromty, toty, val):
    function_type = ir.FunctionType(ir.FloatType(), [ir.IntType(16)])
    instruction = "mov.b32 $0, {0, $1};"
    asm = ir.InlineAsm(function_type, instruction, "=f,h")
    return builder.call(asm, [val])


@intrinsic
def bfloat16_add(typingctx, a, b):
    sig = bfloat16(bfloat16, bfloat16, bfloat16)

    def codegen(context, builder, sig, args):
        i16 = ir.IntType(16)
        function_type = ir.FunctionType(i16, [i16, i16])
        instruction = ("{.reg.b16 one; "
                       "mov.b16 one, 0x3f80U; "
                       "cvt.rn.bf16.f32 $0, $1;}")
        asm = ir.InlineAsm(function_type, instruction, "=h,h,h")
        return builder.call(asm, args)

    return sig, codegen


@overload(operator.add, target='cuda')
def ol_bf16_add(a, b):
    if not (isinstance(a, BFloat16) and isinstance(b, BFloat16)):
        return None

    def impl(a, b):
        return bfloat16_add(a, b)
    return impl


@cuda.jit
def f(x):
    r = bfloat16(types.float32(1.0)) + bfloat16(types.float32(2.0))
    x[()] = types.float32(r)


x = np.array(1, dtype=np.float32)
f[1, 1](x)
print(x[()])
