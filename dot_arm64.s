#include "textflag.h"

TEXT ·dotFloat32sNEON(SB), NOSPLIT, $16-32
	MOVD a+0(FP), R0
	MOVD b+8(FP), R1
	MOVD n+16(FP), R2

	VEOR V0.B16, V0.B16, V0.B16

loop4:
	CMP $4, R2
	BLT tail

	VLD1.P 16(R0), [V1.S4]
	VLD1.P 16(R1), [V2.S4]
	VFMLA V1.S4, V2.S4, V0.S4

	SUB $4, R2, R2
	B loop4

tail:
	VST1 [V0.S4], (RSP)
	FMOVS 0(RSP), F0
	FMOVS 4(RSP), F1
	FADDS F1, F0, F0
	FMOVS 8(RSP), F1
	FADDS F1, F0, F0
	FMOVS 12(RSP), F1
	FADDS F1, F0, F0

	CBZ R2, done

tail_loop:
	FMOVS (R0), F1
	FMOVS (R1), F2
	FMULS F1, F2, F1
	FADDS F1, F0, F0

	ADD $4, R0, R0
	ADD $4, R1, R1
	SUB $1, R2, R2
	CBNZ R2, tail_loop

done:
	FMOVS F0, ret+24(FP)
	RET
