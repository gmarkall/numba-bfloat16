from llvmlite import ir

from numba import cuda, types
from numba.core.types.scalars import Number
from numba.cuda.cudaimpl import registry
from numba.cuda.models import models, register_model

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
def int_to_bfloat16(context, builder, fromty, toty, val):
    function_type = ir.FunctionType(ir.IntType(16), [ir.FloatType()])
    instruction = "cvt.rn.bf16.f32 $0, $1;"
    inst_ir = ir.InlineAsm(function_type, instruction, "=h,h")
    return builder.call(inst_ir, [val])


@cuda.jit
def f():
    bfloat16(types.float32(1.0))


f[1, 1]()
