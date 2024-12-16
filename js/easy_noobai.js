import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

app.registerExtension({
    name: "EasyNoobaiFilter",
    async setup() {
        console.log("EasyNoobaiFilter setup");
    },
    async nodeCreated(node) {
        if (node?.comfyClass === "EasyNoobai") {
            const original_onMouseDown = node.onMouseDown;
            node.onMouseDown = function( e, pos, canvas ) {
                console.log(canvas)
                return original_onMouseDown?.apply(this, arguments);
            }    
        }
    }
});