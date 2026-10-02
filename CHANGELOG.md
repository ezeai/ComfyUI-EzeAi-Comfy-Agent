# Changelog

## 0.1.2
- Memory hand-off: the language model unloads before a render and the image models unload after it, so each step has the GPU and RAM to itself. New setting "Offload between steps" (on by default).
- Autonomous runs now start after the agent has finished its reply, so the language model no longer reloads into VRAM while the render is starting.
- The vision model leaves memory as soon as it has answered.
- llama-server is tied to ComfyUI with a Windows job object and an exit hook, so it cannot be left running (and holding RAM and VRAM) if ComfyUI stops.
- Live VRAM / RAM readout in the mission panel.

## 0.1.1
- Step limit, repeated calls and empty answers now end with an honest short summary and one concrete next step, instead of a generic message.
- Canvas summary lists nodes that are not installed first, with guidance not to rewire them.
- "I prepared a card" with no card is now flagged like any other unverified claim.
- Default step budget raised from 8 to 12.
- Autonomous mission panel: goal, Plan/Build/Run/Review stepper, live render progress, step log, Stop (also interrupts the render).
- No personal paths in shipped defaults.

## 0.1.0
- First public release.
