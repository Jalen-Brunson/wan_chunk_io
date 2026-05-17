import { app } from "../../scripts/app.js";

// Why this exists
// ----------------
// Some workflow JSONs were saved with WanVideoChunkAssembler's
// `audio_offset_sec` widget value as an empty string (""). The Python node
// declares it as FLOAT with default=0.0, but ComfyUI's server-side
// validation (execution.py validate_inputs, ~line 941) does
//
//     if input_type == "FLOAT":
//         val = float(val)
//
// UNCONDITIONALLY for every FLOAT input, BEFORE invoking VALIDATE_INPUTS or
// the node's execute() method. `float("")` raises ValueError, the prompt is
// rejected with `invalid_input_type`, and neither the VALIDATE_INPUTS pass-
// through nor the assemble() try/except can save us — they never run.
//
// The only viable layer to fix at is the frontend, before graphToPrompt
// reads widget values. This extension hooks onConfigure (called when the
// node is restored from a workflow JSON) and onNodeCreated (when a fresh
// node is added), finds the audio_offset_sec widget by name, and replaces
// any non-numeric value with 0.0. After this, the queued prompt carries a
// valid float and the server validates cleanly.
app.registerExtension({
    name: "WanChunkIO.assembler_audio_offset_fix",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanVideoChunkAssembler") return;

        const fixOffset = (node) => {
            const w = node.widgets?.find((w) => w.name === "audio_offset_sec");
            if (!w) return;
            const v = w.value;
            // Already a finite number? Leave it alone.
            if (typeof v === "number" && Number.isFinite(v)) return;
            // String that parses as a float? Coerce to number so ComfyUI's
            // server-side float() succeeds even on locales/edge cases.
            if (typeof v === "string") {
                const trimmed = v.trim();
                if (trimmed !== "") {
                    const parsed = parseFloat(trimmed);
                    if (Number.isFinite(parsed)) {
                        w.value = parsed;
                        return;
                    }
                }
            }
            // Anything else (empty string, null, undefined, NaN) -> default.
            w.value = 0.0;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            fixOffset(this);
            return r;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            fixOffset(this);
            return r;
        };
    },
});
