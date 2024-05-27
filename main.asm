	.file	"common_v1.cpp"
	.text
	.p2align 4
	.globl	_Z4funcPi
	.type	_Z4funcPi, @function
_Z4funcPi:
.LFB1814:
	.cfi_startproc
	movq	$0, (%rdi)
	movq	%rdi, %rax
	leaq	8(%rdi), %rdi
	movq	$0, 4080(%rdi)
	andq	$-8, %rdi
	subq	%rdi, %rax
	leal	4096(%rax), %ecx
	xorl	%eax, %eax
	shrl	$3, %ecx
	rep stosq
	ret
	.cfi_endproc
.LFE1814:
	.size	_Z4funcPi, .-_Z4funcPi
	.section	.text.startup,"ax",@progbits
	.p2align 4
	.type	_GLOBAL__sub_I__Z4funcPi, @function
_GLOBAL__sub_I__Z4funcPi:
.LFB2296:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	leaq	_ZStL8__ioinit(%rip), %rbp
	movq	%rbp, %rdi
	call	_ZNSt8ios_base4InitC1Ev@PLT
	movq	_ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	movq	%rbp, %rsi
	popq	%rbp
	.cfi_def_cfa_offset 8
	leaq	__dso_handle(%rip), %rdx
	jmp	__cxa_atexit@PLT
	.cfi_endproc
.LFE2296:
	.size	_GLOBAL__sub_I__Z4funcPi, .-_GLOBAL__sub_I__Z4funcPi
	.section	.init_array,"aw"
	.align 8
	.quad	_GLOBAL__sub_I__Z4funcPi
	.local	_ZStL8__ioinit
	.comm	_ZStL8__ioinit,1,1
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.4.0-2ubuntu1~18.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
