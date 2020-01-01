; ModuleID = 'simple_func.ll'
source_filename = "simple_func.ll"

; Function Attrs: norecurse nounwind readnone
define i32 @mult(i32 %a, i32 %b) local_unnamed_addr #0 {
  %res = mul i32 %b, %a
  ret i32 %res
}

; Function Attrs: norecurse nounwind readnone
define float @add(float %a, float %b) local_unnamed_addr #0 {
  %res = fadd float %a, %b
  ret float %res
}

; Function Attrs: norecurse nounwind readnone
define i32 @return_first(i32 returned %a, i32 %b) local_unnamed_addr #0 {
start:
  ret i32 %a
}

attributes #0 = { norecurse nounwind readnone }
