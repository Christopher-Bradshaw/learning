	.text
	.file	"simple_func.ll"
	.globl	mult                    # -- Begin function mult
	.p2align	4, 0x90
	.type	mult,@function
mult:                                   # @mult
	.cfi_startproc
# %bb.0:
	imull	%esi, %edi
	movl	%edi, %eax
	retq
.Lfunc_end0:
	.size	mult, .Lfunc_end0-mult
	.cfi_endproc
                                        # -- End function
	.globl	add                     # -- Begin function add
	.p2align	4, 0x90
	.type	add,@function
add:                                    # @add
	.cfi_startproc
# %bb.0:
	addss	%xmm1, %xmm0
	retq
.Lfunc_end1:
	.size	add, .Lfunc_end1-add
	.cfi_endproc
                                        # -- End function
	.globl	return_first            # -- Begin function return_first
	.p2align	4, 0x90
	.type	return_first,@function
return_first:                           # @return_first
	.cfi_startproc
# %bb.0:                                # %start
	movl	%edi, %eax
	retq
.Lfunc_end2:
	.size	return_first, .Lfunc_end2-return_first
	.cfi_endproc
                                        # -- End function

	.section	".note.GNU-stack","",@progbits
