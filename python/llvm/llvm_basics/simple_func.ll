define i32 @mult(i32 %a, i32 %b) {
    %res = mul i32 %a, %b
    ret i32 %res
}

define float @add(float %a, float %b) {
    %res = fadd float %a, %b
    ret float %res
}

define i32 @return_first(i32 %a, i32 %b) {
start:
    ; Because we pass a true statement, we go to left
    ; I want to make this a comparison
    br i1 true, label %left, label %right
left:
    ret i32 %a
right:
    ret i32 %b

}
