# Main frame func in depth

A reminder that this is in `Python/ceval.c:_PyEval_EvalFrameDefault`

## GOTO
Not having written much C goto/labels were a bit confusing. (I was thinking of `goto label` a bit like a function)
* Labels are ignored if you encounter them during normal code flow.
* If you jump to a label you move there and then code just goes on as normal. There is no "going back to where you 'called' the jump". Because goto's are not functions.

## Next opcode
* Usually you don't appear to go all the way back to the top of the infinite loop. You just go to the `fast_next_opcode` label.
* FAST_DISPATCH sends you to the fast_next_opcode. DISPATCH sometimes becomes fast but usually just goes to the top of the loop.
* Note - you never fall out the bottom of the switch statement. Every switch ends with either continue (via DISPATCH sending you back to the top of the loop) or a goto fast_next_opcode/error

## Switch
* The core of this function is a 2k line `switch (opcode)`
* A reminder of how the C switch statement works (see http://lazarenko.me/switch/ for a more complete review)
    * Not entirely sure how any of this works with fall through but I am assuming you never want fall through. You don't want it here...
    * if/else is O(n), switch is somewhere between O(1) - O(log n). Depends how good your hash func is (but it should be perfect because you know all the cases? And there are probably not that many)? Either way, switch is much faster than if/else.
    * Creates jump table - will literally jump you to the right place in the code.
    * However branch prediction is harder for jumps in switch statement. Don't know why but for small switches you just want a if/else.
* Because of questions about performance of the switch they have two ways of doing this (`USE_COMPUTED_GOTOS`).


## GIL
* In the loop there is a chance to drop this thread's lock on the GIL and to let other things run.
