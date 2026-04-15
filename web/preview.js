import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "WanChunkIO.preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "WanVideoChunkAssembler") return;

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const items = message?.gifs;
            if (!items || !items.length) return;
            const v = items[0];
            const params = new URLSearchParams({
                filename: v.filename,
                type: v.type || "output",
                subfolder: v.subfolder || "",
            });
            const url = `/view?${params.toString()}&t=${Date.now()}`;

            if (!this._wanPreviewEl) {
                const el = document.createElement("video");
                el.controls = true;
                el.loop = true;
                el.muted = true;
                el.autoplay = true;
                el.style.width = "100%";
                el.style.maxHeight = "480px";
                el.style.background = "#000";
                this._wanPreviewEl = el;
                this.addDOMWidget("preview", "video", el, {
                    serialize: false,
                    hideOnZoom: false,
                });
                // Reasonable default node size so preview is visible
                if (this.size[1] < 300) this.size[1] = 360;
            }
            this._wanPreviewEl.src = url;
            this._wanPreviewEl.load();
        };
    },
});
