#!/usr/bin/env node
/**
 * Deterministic, self-contained equation SVGs for GitHub README readers that
 * do not implement LaTeX (notably some native mobile app versions).
 *
 * Canonical LaTeX: assets/equations/formulas.json
 * Usage: npm install --no-save --no-package-lock mathjax-full@3.2.2
 *        node scripts/render_equations.cjs
 */
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { mathjax } = require("mathjax-full/js/mathjax.js");
const { TeX } = require("mathjax-full/js/input/tex.js");
const { SVG } = require("mathjax-full/js/output/svg.js");
const { liteAdaptor } = require("mathjax-full/js/adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = require("mathjax-full/js/handlers/html.js");

const root = path.resolve(__dirname, "..");
const dir = path.join(root, "assets", "equations");
const data = JSON.parse(fs.readFileSync(path.join(dir, "formulas.json"), "utf8"));
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({ packages: ["base", "ams", "newcommand"] });
const svg = new SVG({ fontCache: "local" });
const html = mathjax.document("", { InputJax: tex, OutputJax: svg });

function xmlEscape(value) {
  return String(value).replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
for (const { name, label, latex } of data.formulas) {
  if (!/^[a-z0-9-]+$/.test(name)) throw Error("Unsafe equation name: " + name);
  const node = html.convert(latex, { display: true });
  const rendered = adaptor.outerHTML(node);
  const match = rendered.match(/<svg\b[\s\S]*?<\/svg>/);
  if (!match || /<merror\b|data-mjx-error/.test(rendered)) {
    throw Error("MathJax failed to render: " + name);
  }
  let image = match[0];
  image = image.replace(/style="[^"]*"/, 'style="color:#172b46;background-color:#ffffff"');
  image = image.replace("<svg ", '<svg aria-label="' + xmlEscape(label) + '" ');
  image = image.replace(/(<svg\b[^>]*>)/, '$1<title>' + xmlEscape(label) + '</title>');
  const target = path.join(dir, name + ".svg");
  fs.writeFileSync(target, image + "\n", "utf8");
  console.log("Rendered", path.relative(root, target), image.length, "bytes");
}
