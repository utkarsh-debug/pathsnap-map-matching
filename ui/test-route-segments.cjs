// Run: node ui/test-route-segments.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const code = fs.readFileSync("ui/app.js", "utf8");
const scope = {};
vm.createContext(scope);
vm.runInContext(code.slice(code.indexOf("function validLatLon"), code.indexOf("function formatPercent")), scope);
const result = scope.projectRoute([], [], [
  [[1, 1], [1, 2]], [[3, 3], [3, 4]]
]);
assert.equal(result.segments.length, 2);
assert.equal(result.route.length, 4);
const svgPath = result.segments.map(s => scope.pathData(s, result.toScreen)).join(" ");
assert.equal((svgPath.match(/M/g) || []).length, 2, "each segment must start a separate SVG subpath");
assert.equal((svgPath.match(/L/g) || []).length, 2, "no extra connector between segments");
const legacy = scope.projectRoute([], [[1, 1], [1, 2]]);
assert.equal(legacy.segments.length, 1, "old API format remains supported");
assert.equal(scope.projectRoute([], [], []), null);
console.log("UI geometry checks passed: separate subpaths, legacy response, empty route.");
