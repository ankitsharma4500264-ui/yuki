import React from "react";

// Minimal, dependency-free markdown -> React elements. Never uses dangerouslySetInnerHTML, so model/document text
// can't inject HTML. Supports: headings, lists, quotes, code, tables, **bold**, *italic*, `code`, and [S1] citation chips.
const INLINE_SRC = /(\*\*[^*\n]+\*\*|\*(?! )[^*\n]*[^* \n]\*|`[^`\n]+`|\[S\d+(?:, ?S\d+)*\])/;

function inline(text, key) {
  const out = []; let last = 0, m, i = 0;
  const re = new RegExp(INLINE_SRC.source, "g");   // a fresh regex per call: nested calls must not share lastIndex
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const t = m[0], k = `${key}-${i++}`;
    if (t.startsWith("**")) out.push(<strong key={k}>{inline(t.slice(2, -2), k)}</strong>);
    else if (t.startsWith("`")) out.push(<code key={k}>{t.slice(1, -1)}</code>);
    else if (t.startsWith("[S")) t.slice(1, -1).split(/, ?/).forEach((l) => out.push(<a key={k + l} className="ya-chip" href={`#ya-src-${l}`}>{l}</a>));
    else out.push(<em key={k}>{t.slice(1, -1)}</em>);
    last = m.index + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

const cells = (line) => line.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
const isSep = (line) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line) && line.includes("-");
const LIST = /^\s*(?:[-*\u2022]|\d+[.)])\s+(.*)$/;

export function Rich({ text }) {
  const lines = (text || "").replace(/\r/g, "").split("\n");
  const blocks = []; let i = 0, n = 0;
  while (i < lines.length) {
    const line = lines[i], key = "b" + n++;
    if (/^```/.test(line)) {                                   // fenced code
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++; blocks.push(<pre key={key}><code>{buf.join("\n")}</code></pre>); continue;
    }
    if (line.trim() === "") { i++; continue; }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) { blocks.push(<h4 key={key} className="ya-h">{inline(h[2], key)}</h4>); i++; continue; }
    if (line.trim().startsWith("|") && i + 1 < lines.length && isSep(lines[i + 1])) {   // table
      const head = cells(line); i += 2; const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(cells(lines[i++]));
      blocks.push(<table key={key}><thead><tr>{head.map((c, j) => <th key={j}>{inline(c, key + "h" + j)}</th>)}</tr></thead>
        <tbody>{rows.map((r, a) => <tr key={a}>{r.map((c, j) => <td key={j}>{inline(c, `${key}r${a}c${j}`)}</td>)}</tr>)}</tbody></table>);
      continue;
    }
    if (LIST.test(line)) {                                      // list
      const ordered = /^\s*\d/.test(line), items = [];
      while (i < lines.length && LIST.test(lines[i])) items.push(LIST.exec(lines[i++])[1]);
      const Tag = ordered ? "ol" : "ul";
      blocks.push(<Tag key={key}>{items.map((t, j) => <li key={j}>{inline(t, `${key}i${j}`)}</li>)}</Tag>); continue;
    }
    if (/^>\s?/.test(line)) {
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ""));
      blocks.push(<blockquote key={key}>{inline(buf.join("\n"), key)}</blockquote>); continue;
    }
    const buf = [];                                             // paragraph
    while (i < lines.length && lines[i].trim() !== "" && !/^(```|#{1,4}\s|>\s?)/.test(lines[i]) && !LIST.test(lines[i]) &&
           !(lines[i].trim().startsWith("|") && i + 1 < lines.length && isSep(lines[i + 1]))) buf.push(lines[i++]);
    if (buf.length === 0) buf.push(lines[i++]);       // always make progress
    blocks.push(<p key={key}>{inline(buf.join("\n"), key)}</p>);
  }
  return <div className="ya-md">{blocks}</div>;
}
