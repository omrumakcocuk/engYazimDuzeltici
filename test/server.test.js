const test = require("node:test");
const assert = require("node:assert/strict");

const {
  attachOcrAnchors,
  buildCorrectionTranscript,
  buildSentenceGroups,
  correctionsFromGeminiEdits,
  detectAdvancedGrammar,
  detectCapitalizationErrors,
  detectDeterministicGrammar,
  diffLineToCorrections,
  enforceParallelCorrectionForms,
  filterLikelyOcrArtifacts,
  filterProtectedProperNames,
  filterUnrenderableCorrections,
  groupOcrWordsIntoLines,
  isSafeCorrection,
  mergeCorrections,
  mergeMisplitSentenceGroups,
  normalizeGeminiEdits,
  normalizeGeminiPunctuation,
  normalizeOcrNumber,
  organizeSentenceGroupsWithGemini,
  parseGeminiJson,
  parseGoogleVisionWords,
  selectGoogleHandwrittenWords,
  selectHandwrittenWords,
  trimRewriteToChangedSpan
} = require("../server");

function sentence(texts) {
  return [{
    id: "group_1",
    words: texts.map((text, index) => word(`s${index}`, text, index * 80)),
    text: texts.join(" ")
  }];
}

test("Gemini JSON parser repairs fenced responses and trailing commas", () => {
  assert.deepEqual(parseGeminiJson(`\`\`\`json
  {"corrected":"Hello my name is Ahmet", "edits": [],}
  \`\`\``), {
    corrected: "Hello my name is Ahmet",
    edits: []
  });
});

test("an incomplete Gemini edit falls back to corrected-sentence alignment without retrying", () => {
  const group = sentence(["My", "name", "Ahmet"])[0];
  assert.deepEqual(normalizeGeminiEdits(group, [{
    target_ids: [], left_id: "s1", right_id: "s2"
  }]), []);
  const corrections = correctionsFromGeminiEdits(group, [], "My name is Ahmet")
    || diffLineToCorrections(group, "My name is Ahmet");
  assert.equal(corrections.length, 1);
  assert.equal(corrections[0].action, "insert");
  assert.equal(corrections[0].replacement, "is");
});

test("inpainting anchors are clipped between neighboring text-line centers", () => {
  const words = [
    { id: "upper", text: "wrong", x: 100, y: 100, width: 90, height: 60 },
    { id: "lower", text: "keep", x: 100, y: 220, width: 80, height: 30 }
  ];
  const [correction] = attachOcrAnchors([{
    action: "replace", original: "wrong", replacement: "right",
    target_id: "upper", left_id: "", right_id: ""
  }], words);
  assert.equal(correction.anchors[0].safeTop, 0);
  assert.ok(correction.anchors[0].safeBottom > 170);
  assert.ok(correction.anchors[0].safeBottom < 200);
  assert.equal(correction.anchors[0].lineBaseline, 160);
  assert.equal(correction.anchors[0].lineHeight, 60);
});

test("replacement anchors may use neighboring whitespace without touching words", () => {
  const words = [
    { id: "left", text: "we", x: 100, y: 100, width: 40, height: 40 },
    { id: "target", text: "go", x: 200, y: 100, width: 35, height: 40 },
    { id: "right", text: "to", x: 300, y: 100, width: 30, height: 40 }
  ];
  const [correction] = attachOcrAnchors([{
    action: "replace", original: "go", replacement: "went",
    target_id: "target", left_id: "", right_id: ""
  }], words);
  const anchor = correction.anchors[0];
  assert.equal(anchor.slotX, 170);
  assert.equal(anchor.slotWidth, 97.5);
  assert.ok(anchor.slotX > words[0].x + words[0].width);
  assert.ok(anchor.slotX + anchor.slotWidth < words[2].x);
});

test("correction anchors use the target word baseline on a curved handwritten line", () => {
  const words = [
    { id: "left", text: "we", x: 100, y: 120, width: 40, height: 35, layoutLine: "curve" },
    { id: "target", text: "go", x: 200, y: 105, width: 45, height: 35, layoutLine: "curve" },
    { id: "right", text: "home", x: 310, y: 115, width: 80, height: 35, layoutLine: "curve" }
  ];
  const [correction] = attachOcrAnchors([{
    action: "replace", original: "go", replacement: "went",
    target_id: "target", left_id: "", right_id: ""
  }], words);
  assert.equal(correction.anchors[0].lineBaseline, 140);
  assert.equal(correction.anchors[0].baselineSlope, undefined);
});

function word(id, text, x, y = 100) {
  return { id, text, x, y, width: 60, height: 30 };
}

test("Google Vision word polygons are normalized to the shared coordinate space", () => {
  const words = parseGoogleVisionWords({
    fullTextAnnotation: {
      pages: [{
        width: 2000,
        height: 1000,
        blocks: [{ paragraphs: [{ words: [{
          confidence: 0.94,
          symbols: [{ text: "H" }, { text: "i" }],
          boundingBox: {
            vertices: [{ x: 200, y: 100 }, { x: 500, y: 100 }, { x: 500, y: 200 }, { x: 200, y: 200 }],
            normalizedVertices: []
          }
        }] }] }]
      }]
    }
  });
  assert.deepEqual(words, [{
    id: "w0", text: "Hi", x: 100, y: 100, width: 150, height: 100, confidence: 0.94
  }]);
});

function visionWord(text, left = 0, symbols) {
  return {
    confidence: 0.9,
    symbols: symbols || text.split("").map((character) => ({ text: character })),
    boundingBox: {
      vertices: [{ x: left, y: 0 }, { x: left + 10, y: 0 }, { x: left + 10, y: 10 }, { x: left, y: 10 }],
      normalizedVertices: []
    }
  };
}

test("trailing punctuation attached to a word is captured separately from its text", () => {
  const words = parseGoogleVisionWords({
    fullTextAnnotation: { pages: [{ width: 100, height: 100, blocks: [{ paragraphs: [{ words: [
      visionWord("cats,", 0),
      visionWord("Ankara.", 20)
    ] }] }] }] }
  });
  assert.equal(words[0].text, "cats");
  assert.equal(words[0].punct, ",");
  assert.equal(words[1].text, "Ankara");
  assert.equal(words[1].punct, ".");
});

test("a standalone punctuation symbol attaches to its nearest word by position", () => {
  const words = parseGoogleVisionWords({
    fullTextAnnotation: { pages: [{ width: 100, height: 100, blocks: [{ paragraphs: [{ words: [
      // Vision's raw array order does not match reading order here (the
      // mark is listed second even though "Ahmet" sits further right) -
      // position, not array order, must decide which word it attaches to.
      visionWord("?", 30),
      visionWord("Ahmet", 10)
    ] }] }] }] }
  });
  assert.equal(words.length, 1);
  assert.equal(words[0].text, "Ahmet");
  assert.equal(words[0].punct, "?");
});

test("a standalone punctuation symbol on a different physical line is not attached", () => {
  const words = parseGoogleVisionWords({
    fullTextAnnotation: { pages: [{ width: 100, height: 100, blocks: [{ paragraphs: [{ words: [
      visionWord("Ahmet", 10),
      { ...visionWord(".", 10), boundingBox: {
        vertices: [{ x: 10, y: 200 }, { x: 20, y: 200 }, { x: 20, y: 210 }, { x: 10, y: 210 }], normalizedVertices: []
      } }
    ] }] }] }] }
  });
  assert.equal(words.length, 1);
  assert.equal(words[0].punct, undefined);
});

test("three or more dots normalize to an ellipsis mark", () => {
  const words = parseGoogleVisionWords({
    fullTextAnnotation: { pages: [{ width: 100, height: 100, blocks: [{ paragraphs: [{ words: [
      visionWord("wait..."),
    ] }] }] }] }
  });
  assert.equal(words[0].text, "wait");
  assert.equal(words[0].punct, "...");
});

test("normalizeGeminiPunctuation keeps only entries with a known word id and an allowed mark", () => {
  const group = sentence(["cats", "and", "dogs"])[0];
  const result = normalizeGeminiPunctuation(group, [
    { after_id: "s0", mark: "!" },
    { after_id: "unknown-id", mark: "." },
    { after_id: "s2", mark: ";" },
    { after_id: "s2", mark: "." }
  ]);
  assert.deepEqual(result, [
    { after_id: "s2", mark: "." }
  ]);
});

test("normalizeGeminiPunctuation never lets a comma through, even mid-sentence", () => {
  const group = sentence(["cats", "and", "dogs"])[0];
  const result = normalizeGeminiPunctuation(group, [
    { after_id: "s0", mark: "," }
  ]);
  assert.deepEqual(result, []);
});

test("normalizeGeminiPunctuation drops a sentence-ending mark placed mid-group even when it also proposes a comma there", () => {
  const group = sentence(["I", "should", "pay", "more", "attention", "to", "grammar"])[0];
  const result = normalizeGeminiPunctuation(group, [
    { after_id: "s4", mark: "." },
    { after_id: "s1", mark: "," },
    { after_id: "s6", mark: "." }
  ]);
  assert.deepEqual(result, [
    { after_id: "s6", mark: "." }
  ]);
});

test("the typed transcript adds a missing comma without altering an unrelated word correction", () => {
  const group = sentence(["My", "sister", "enjoy", "cats", "and", "dogs"])[0];
  group.words[3].punct = "";
  const corrections = [
    { action: "replace", original: "enjoy", replacement: "enjoys", target_id: "s2", left_id: "", right_id: "" }
  ];
  const punctuationByWordId = new Map([["s3", ","]]);
  const transcript = buildCorrectionTranscript([group], corrections, punctuationByWordId);
  assert.deepEqual(transcript, [
    { text: "My", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "sister", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "enjoys", changed: true, reason: "", punct: "", punctChanged: false },
    { text: "cats", changed: false, reason: "", punct: ",", punctChanged: true },
    { text: "and", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "dogs", changed: false, reason: "", punct: "", punctChanged: false }
  ]);
});

test("the typed transcript leaves already-correct punctuation alone", () => {
  const group = sentence(["Hello", "Ahmet"])[0];
  group.words[1].punct = ".";
  const punctuationByWordId = new Map([["s1", "."]]);
  const transcript = buildCorrectionTranscript([group], [], punctuationByWordId);
  assert.deepEqual(transcript, [
    { text: "Hello", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "Ahmet", changed: false, reason: "", punct: ".", punctChanged: false }
  ]);
});

test("a punctuation-only correction colors only the mark, not the word it follows", () => {
  const group = sentence(["Hello", "Ahmet"])[0];
  const punctuationByWordId = new Map([["s1", "."]]);
  const transcript = buildCorrectionTranscript([group], [], punctuationByWordId);
  assert.deepEqual(transcript, [
    { text: "Hello", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "Ahmet", changed: false, reason: "", punct: ".", punctChanged: true }
  ]);
});

test("a word's own explanation survives a punctuation change landing on the same token", () => {
  const group = sentence(["My", "sister", "enjoy", "cats"])[0];
  const corrections = [{
    action: "replace", original: "enjoy", replacement: "enjoys",
    reason: "Third-person singular subjects need an -s ending.",
    target_id: "s2", left_id: "", right_id: ""
  }];
  const punctuationByWordId = new Map([["s3", "."]]);
  const transcript = buildCorrectionTranscript([group], corrections, punctuationByWordId);
  const last = transcript[transcript.length - 1];
  assert.equal(last.text, "cats");
  assert.equal(last.changed, false);
  assert.equal(last.punct, ".");
  assert.equal(last.punctChanged, true);
  assert.equal(last.reason, "");
  const enjoysToken = transcript.find((token) => token.text === "enjoys");
  assert.equal(enjoysToken.reason, "Third-person singular subjects need an -s ending.");
});

test("Gemini uses coordinate layout before visual fallback", async () => {
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    requests.push(request);
    return {
      ok: true,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: JSON.stringify({
          lines: [{ word_ids: ["w1", "w2"] }],
          sentences: [{ word_ids: ["w1", "w2"] }]
        }) }] } }]
      })
    };
  };
  try {
    const result = await organizeSentenceGroupsWithGemini(
      "data:image/jpeg;base64,AA==",
      [word("w1", "She", 100), word("w2", "reads", 180)]
    );
    assert.equal(result.source, "coordinates");
    assert.equal(requests.length, 1);
    assert.equal(requests[0].contents[0].parts.length, 1);
  } finally {
    global.fetch = originalFetch;
  }
});

test("a sentence's word order is corrected from physical-line position when Gemini swaps two word IDs", async () => {
  const originalFetch = global.fetch;
  global.fetch = async (_url, options) => {
    const request = JSON.parse(options.body);
    const wordsRequested = request.contents[0].parts.length === 1;
    return {
      ok: true,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: JSON.stringify({
          lines: [{ word_ids: ["w1", "w2", "w3"] }, { word_ids: ["w4"] }],
          // A real case: the last word of line 1 ("all") and the sole word
          // wrapped onto line 2 ("day") arrive swapped in the sentence's
          // word_id sequence, even though the lines array (above) correctly
          // places w3 on line 1 and w4 on line 2.
          sentences: [{ word_ids: ["w1", "w2", "w4", "w3"] }]
        }) }] } }]
      })
    };
  };
  try {
    const result = await organizeSentenceGroupsWithGemini(
      "data:image/jpeg;base64,AA==",
      [
        word("w1", "it", 100, 100), word("w2", "rains", 180, 100), word("w3", "all", 260, 100),
        word("w4", "day", 100, 200)
      ]
    );
    assert.equal(result.groups[0].text, "it rains all day");
  } finally {
    global.fetch = originalFetch;
  }
});

// Note: a follow-up attempt replaced Gemini's own physical-line assignment
// with pure y-coordinate clustering here, reasoning that geometry can never
// be wrong. In practice this repo's line clustering has no curvature/slant
// compensation wired in (see groupOcrWordsIntoLines - the correction for
// that exists but is disabled after an earlier, unrelated regression), so
// on any photographed page with real-world curl or slant it misclusters
// words into the wrong physical line far more often and far worse than
// Gemini's own occasional line-assignment mistake. That attempt was
// reverted; Gemini's line assignment remains the trusted source for now.

test("an incomplete coordinate layout retries with the photograph", async () => {
  const originalFetch = global.fetch;
  let requestCount = 0;
  global.fetch = async (_url, options) => {
    requestCount += 1;
    const request = JSON.parse(options.body);
    const wordIds = request.contents[0].parts.length === 1 ? ["w1"] : ["w1", "w2"];
    return {
      ok: true,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: JSON.stringify({
          lines: [{ word_ids: wordIds }],
          sentences: [{ word_ids: wordIds }]
        }) }] } }]
      })
    };
  };
  try {
    const result = await organizeSentenceGroupsWithGemini(
      "data:image/jpeg;base64,AA==",
      [word("w1", "She", 100), word("w2", "reads", 180)]
    );
    assert.equal(result.source, "visual");
    assert.equal(requestCount, 2);
  } finally {
    global.fetch = originalFetch;
  }
});

test("a duplicate or invented Gemini layout ID does not discard valid OCR layout", async () => {
  const originalFetch = global.fetch;
  let requestCount = 0;
  global.fetch = async () => {
    requestCount += 1;
    return {
      ok: true,
      json: async () => ({
        candidates: [{ content: { parts: [{ text: JSON.stringify({
          lines: [{ word_ids: ["w1", "imaginary", "w2", "w2"] }],
          sentences: [{ word_ids: ["w1", "w2", "imaginary"] }]
        }) }] } }]
      })
    };
  };
  try {
    const result = await organizeSentenceGroupsWithGemini(
      "data:image/jpeg;base64,AA==",
      [word("w1", "She", 100), word("w2", "reads", 180)]
    );
    assert.equal(result.source, "coordinates");
    assert.equal(requestCount, 1);
    assert.deepEqual(result.groups[0].words.map((item) => item.id), ["w1", "w2"]);
  } finally {
    global.fetch = originalFetch;
  }
});

test("a full punctuation-free line can wrap before a lowercase continuation", () => {
  const lines = [
    { id: "line_1", words: [word("w1", "My", 80), word("w2", "brother", 180), word("w3", "said", 320), word("w4", "he", 720)] },
    { id: "line_2", words: [word("w5", "wanted", 80, 160), word("w6", "to", 200, 160), word("w7", "leave", 280, 160)] }
  ];
  assert.equal(buildSentenceGroups(lines).length, 1);
});

test("Google OCR ignores keyboard rows before selecting handwriting", () => {
  const words = [
    word("k1", "A", 10, 20), word("k2", "S", 80, 20), word("k3", "D", 150, 20),
    word("w1", "my", 10, 400), word("w2", "sister", 90, 400), word("w3", "reads", 190, 400),
    word("w4", "she", 10, 470), word("w5", "has", 90, 470), word("w6", "books", 170, 470)
  ];
  assert.deepEqual(
    selectGoogleHandwrittenWords(words).map((item) => item.id),
    ["w1", "w2", "w3", "w4", "w5", "w6"]
  );
});

test("a one-word wrapped ending remains part of the handwritten document", () => {
  const words = [
    word("w1", "She", 120, 100), word("w2", "finished", 220, 100),
    word("w3", "the", 400, 100), word("w4", "day", 500, 100),
    word("w5", "before", 120, 155)
  ];
  assert.deepEqual(selectGoogleHandwrittenWords(words).map((item) => item.text), [
    "She", "finished", "the", "day", "before"
  ]);
});

test("perspective does not discard smaller handwriting at the top of a long letter", () => {
  const words = [
    { id: "top1", text: "She", x: 100, y: 100, width: 45, height: 13 },
    { id: "top2", text: "have", x: 160, y: 100, width: 55, height: 13 },
    { id: "top3", text: "cats", x: 230, y: 100, width: 45, height: 13 },
    { id: "bottom1", text: "They", x: 100, y: 700, width: 90, height: 42 },
    { id: "bottom2", text: "study", x: 210, y: 700, width: 110, height: 42 },
    { id: "bottom3", text: "hard", x: 340, y: 700, width: 90, height: 42 }
  ];
  assert.deepEqual(
    selectGoogleHandwrittenWords(words).map((item) => item.id),
    words.map((item) => item.id)
  );
});

test("slanted handwriting remains grouped into complete visual lines", () => {
  const words = [
    { id: "w0", text: "Likes", x: 70, y: 210, width: 207, height: 226 },
    { id: "w1", text: "she", x: 327, y: 171, width: 135, height: 210 },
    { id: "w2", text: "football", x: 515, y: 106, width: 254, height: 236 },
    { id: "w3", text: "His", x: 85, y: 539, width: 125, height: 226 },
    { id: "w4", text: "name", x: 252, y: 486, width: 176, height: 238 },
    { id: "w5", text: "is", x: 460, y: 456, width: 81, height: 216 },
    { id: "w6", text: "john", x: 563, y: 406, width: 184, height: 241 },
    { id: "w7", text: "Cena", x: 762, y: 359, width: 182, height: 239 }
  ];
  assert.equal(groupOcrWordsIntoLines(words).length, 2);
  assert.deepEqual(
    selectGoogleHandwrittenWords(words).map((item) => item.id).sort(),
    words.map((item) => item.id).sort()
  );
});

test("a pure word-order error changes only the words that moved", () => {
  const line = {
    id: "group_1",
    lineIds: ["line_1"],
    words: [word("w1", "Likes", 10), word("w2", "She", 90), word("w3", "football", 160)]
  };
  assert.deepEqual(
    diffLineToCorrections(line, "She likes football").map(({ original, replacement, target_id }) => ({
      original, replacement, target_id
    })),
    [
      { original: "Likes", replacement: "She", target_id: "w1" },
      { original: "She", replacement: "likes", target_id: "w2" }
    ]
  );
});

test("a fronted lexical verb is deterministically reordered before a pronoun subject", () => {
  const words = [word("w1", "Likes", 10), word("w2", "She", 90), word("w3", "football", 160)];
  const corrections = detectDeterministicGrammar(words);
  assert.deepEqual(
    corrections.map(({ original, replacement, target_id }) => ({ original, replacement, target_id })),
    [
      { original: "Likes", replacement: "She", target_id: "w1" },
      { original: "She", replacement: "likes", target_id: "w2" }
    ]
  );
  assert.equal(corrections.some((item) => item.target_id === "w3"), false);
});

test("the typed transcript drops wrong words entirely and marks only the correction red", () => {
  const group = sentence(["My", "brother", "enjoy", "read", "books"])[0];
  const corrections = [
    { action: "replace", original: "enjoy", replacement: "enjoys", target_id: "s2", left_id: "", right_id: "" },
    { action: "insert", original: "", replacement: "to", reason: "", target_id: "", left_id: "s3", right_id: "s4" }
  ];
  const transcript = buildCorrectionTranscript([group], corrections);
  assert.deepEqual(transcript, [
    { text: "My", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "brother", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "enjoys", changed: true, reason: "", punct: "", punctChanged: false },
    { text: "read", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "to", changed: true, reason: "", punct: "", punctChanged: false },
    { text: "books", changed: false, reason: "", punct: "", punctChanged: false }
  ]);
});

test("the typed transcript drops a pure word deletion instead of leaving a blank token", () => {
  const group = sentence(["Although", "it", "was", "hot", "but", "everyone", "smiled"])[0];
  const corrections = [
    { action: "replace", original: "but", replacement: "", target_id: "s4", left_id: "", right_id: "" }
  ];
  const transcript = buildCorrectionTranscript([group], corrections);
  assert.deepEqual(transcript.map((token) => token.text), ["Although", "it", "was", "hot", "everyone", "smiled"]);
  assert.equal(transcript.some((token) => token.text === ""), false);
});

test("the typed transcript collapses a rewrite_line's whole phrase into one red replacement", () => {
  const group = sentence(["My", "brother", "enjoy", "to", "read", "books"])[0];
  const corrections = [{
    action: "rewrite_line", original: "enjoy to read", replacement: "enjoys reading",
    target_ids: ["s2", "s3", "s4"], target_id: "", left_id: "", right_id: ""
  }];
  const transcript = buildCorrectionTranscript([group], corrections);
  assert.deepEqual(transcript, [
    { text: "My", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "brother", changed: false, reason: "", punct: "", punctChanged: false },
    { text: "enjoys reading", changed: true, reason: "", punct: "", punctChanged: false },
    { text: "books", changed: false, reason: "", punct: "", punctChanged: false }
  ]);
});

test("mergeMisplitSentenceGroups joins two groups when the split has neither terminal punctuation nor a capital start", () => {
  const groupA = { id: "group_1", lineIds: ["l1"], words: [word("w1", "today", 10)], text: "today" };
  const groupB = { id: "group_2", lineIds: ["l1"], words: [word("w2", "and", 90)], text: "and" };
  const result = mergeMisplitSentenceGroups([groupA, groupB]);
  assert.equal(result.length, 1);
  assert.deepEqual(result[0].words.map((w) => w.id), ["w1", "w2"]);
  assert.equal(result[0].text, "today and");
});

test("mergeMisplitSentenceGroups keeps groups separate when the first ends with terminal punctuation", () => {
  const groupA = { id: "group_1", lineIds: ["l1"], words: [{ ...word("w1", "today", 10), punct: "." }], text: "today" };
  const groupB = { id: "group_2", lineIds: ["l1"], words: [word("w2", "and", 90)], text: "and" };
  const result = mergeMisplitSentenceGroups([groupA, groupB]);
  assert.equal(result.length, 2);
});

test("mergeMisplitSentenceGroups keeps groups separate when the next one starts with a capital letter", () => {
  const groupA = { id: "group_1", lineIds: ["l1"], words: [word("w1", "today", 10)], text: "today" };
  const groupB = { id: "group_2", lineIds: ["l1"], words: [word("w2", "She", 90)], text: "She" };
  const result = mergeMisplitSentenceGroups([groupA, groupB]);
  assert.equal(result.length, 2);
});

test("a redundant more before a comparative adjective is merged into one rewrite", () => {
  const words = [
    word("w1", "It", 10), word("w2", "was", 80), word("w3", "more", 150), word("w4", "better", 230)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.action === "rewrite_line");
  assert.ok(correction);
  assert.deepEqual(correction.target_ids, ["w3", "w4"]);
  assert.equal(correction.replacement, "better");
});

test("more before an unrelated -er word is never touched", () => {
  const words = [
    word("w1", "I", 10), word("w2", "like", 80), word("w3", "more", 150), word("w4", "water", 230)
  ];
  assert.equal(
    detectDeterministicGrammar(words).some((item) => item.action === "rewrite_line"),
    false
  );
});

test("a present-perfect auxiliary before a simple-past-only verb is dropped", () => {
  const words = [
    word("w1", "I", 10), word("w2", "have", 80), word("w3", "saw", 150), word("w4", "many", 230), word("w5", "birds", 310)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.action === "rewrite_line");
  assert.ok(correction);
  assert.deepEqual(correction.target_ids, ["w2", "w3"]);
  assert.equal(correction.replacement, "saw");
});

test("have/has/had before a verb whose participle matches the past tense is left alone", () => {
  const words = [
    word("w1", "She", 10), word("w2", "has", 80), word("w3", "made", 150), word("w4", "a", 230), word("w5", "cake", 310)
  ];
  assert.equal(
    detectDeterministicGrammar(words).some((item) => item.action === "rewrite_line"),
    false
  );
});

test("a rewrite_line carrying an unchanged edge word for a pure deletion shrinks to a single-word replace", () => {
  const correction = {
    action: "rewrite_line", original: "was grew", replacement: "grew",
    reason: "", target_ids: ["w1", "w2"], target_id: "", left_id: "", right_id: ""
  };
  assert.deepEqual(trimRewriteToChangedSpan(correction), {
    action: "replace", original: "was", replacement: "", reason: "", target_id: "w1", left_id: "", right_id: ""
  });
});

test("a rewrite_line with unchanged words on both edges shrinks to just the changed middle word", () => {
  const correction = {
    action: "rewrite_line", original: "the old book", replacement: "the new book",
    reason: "", target_ids: ["w1", "w2", "w3"], target_id: "", left_id: "", right_id: ""
  };
  assert.deepEqual(trimRewriteToChangedSpan(correction), {
    action: "replace", original: "old", replacement: "new", reason: "", target_id: "w2", left_id: "", right_id: ""
  });
});

test("a rewrite_line with no unchanged edge words is left untouched", () => {
  const correction = {
    action: "rewrite_line", original: "enjoy to read", replacement: "enjoys reading",
    reason: "", target_ids: ["w1", "w2", "w3"], target_id: "", left_id: "", right_id: ""
  };
  assert.deepEqual(trimRewriteToChangedSpan(correction), correction);
});

test("trimRewriteToChangedSpan leaves non-rewrite_line corrections alone", () => {
  const correction = { action: "replace", original: "was", replacement: "were", target_id: "w1", left_id: "", right_id: "" };
  assert.deepEqual(trimRewriteToChangedSpan(correction), correction);
});

test("a fronted lexical verb with a noun-phrase subject is reordered", () => {
  const words = [
    word("w1", "Enjoys", 10), word("w2", "my", 90), word("w3", "father", 160),
    word("w4", "fishing", 240), word("w5", "every", 320), word("w6", "weekend", 400)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.action === "rewrite_line");
  assert.ok(correction);
  assert.deepEqual(correction.target_ids, ["w1", "w2", "w3"]);
  assert.equal(correction.replacement, "My father enjoys");
});

test("a fronted lexical verb followed by an object noun phrase is left alone", () => {
  const words = [
    word("w1", "He", 10), word("w2", "enjoys", 90), word("w3", "the", 160),
    word("w4", "reptile", 240), word("w5", "house", 320)
  ];
  assert.equal(
    detectDeterministicGrammar(words).some((item) => item.action === "rewrite_line"),
    false
  );
});

test("a word-order error is found after another clause on the same physical line", () => {
  const words = [
    word("w1", "they", 10), word("w2", "study", 80), word("w3", "hard", 150),
    word("w4", "Likes", 230), word("w5", "he", 310), word("w6", "football", 370)
  ];
  const byTarget = new Map(detectDeterministicGrammar(words).map((item) => [item.target_id, item.replacement]));
  assert.equal(byTarget.get("w4"), "He");
  assert.equal(byTarget.get("w5"), "likes");
  assert.equal(byTarget.has("w6"), false);
});

test("a past marker tolerates one-character OCR noise in an irregular verb", () => {
  const words = [
    word("w1", "Last", 10), word("w2", "month", 80), word("w3", "they", 160),
    word("w4", "buty", 230), word("w5", "three", 300), word("w6", "books", 380)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w4");
  assert.equal(correction.replacement, "bought");
});

test("past-tense OCR tolerance does not reinterpret nearby function words or pronouns", () => {
  const words = [
    word("w1", "Yesterday", 10), word("w2", "she", 100),
    word("w3", "go", 170), word("w4", "to", 230), word("w5", "school", 290)
  ];
  const byTarget = new Map(detectDeterministicGrammar(words).map((item) => [item.target_id, item.replacement]));
  assert.equal(byTarget.get("w3"), "went");
  assert.equal(byTarget.has("w2"), false);
  assert.equal(byTarget.has("w4"), false);
});

test("a content word is never changed into a nearby function word by fuzzy matching", () => {
  assert.equal(isSafeCorrection({ action: "replace", original: "go", replacement: "to" }), false);
});

test("long letters retain line-local high-confidence grammar checks", () => {
  const words = [
    word("w1", "She", 10, 100), word("w2", "have", 80, 100), word("w3", "two", 150, 100),
    word("w4", "cat", 220, 100), word("w5", "Yesterday", 300, 100), word("w6", "we", 390, 100),
    word("w7", "go", 450, 100), word("w8", "to", 510, 100), word("w9", "school", 570, 100),
    word("w10", "My", 10, 200), word("w11", "friend", 80, 200), word("w12", "doesn't", 160, 200),
    word("w13", "enjoys", 240, 200), word("w14", "going", 320, 200), word("w15", "to", 390, 200),
    word("w16", "library", 450, 200),
    word("w17", "Their", 10, 300), word("w18", "teacher", 90, 300), word("w19", "were", 180, 300),
    word("w20", "happy", 250, 300),
    word("w21", "Last", 10, 400), word("w22", "month", 80, 400), word("w23", "they", 160, 400),
    word("w24", "buy", 230, 400), word("w25", "three", 300, 400), word("w26", "book", 380, 400)
  ];
  const byTarget = new Map(detectDeterministicGrammar(words).map((item) => [item.target_id, item.replacement]));
  assert.equal(byTarget.get("w2"), "has");
  assert.equal(byTarget.get("w4"), "cats");
  assert.equal(byTarget.get("w7"), "went");
  assert.equal(byTarget.get("w13"), "enjoy");
  assert.equal(byTarget.get("w19"), "was");
  assert.equal(byTarget.get("w24"), "bought");
  assert.equal(byTarget.get("w26"), "books");
});

test("a misspelled negative auxiliary still forces the following base verb", () => {
  const words = [
    word("w1", "My", 10), word("w2", "sister", 80),
    word("w3", "dont", 160), word("w4", "likes", 230), word("w5", "reading", 300)
  ];
  const byTarget = new Map(detectDeterministicGrammar(words).map((item) => [item.target_id, item.replacement]));
  assert.equal(byTarget.get("w3"), "doesn't");
  assert.equal(byTarget.get("w4"), "like");
});

test("a preference rule never pluralizes a similar word on the next physical line", () => {
  const words = [
    word("w1", "Likes", 10, 100), word("w2", "She", 90, 100), word("w3", "football", 160, 100),
    word("w4", "His", 10, 220), word("w5", "name", 90, 220), word("w6", "is", 160, 220), word("w7", "John", 210, 220)
  ];
  assert.equal(
    detectDeterministicGrammar(words).some((item) => item.target_id === "w5"),
    false
  );
});

test("short unrelated words are rejected as unsafe spelling corrections", () => {
  assert.equal(isSafeCorrection({ action: "replace", original: "name", replacement: "games" }), false);
});

test("a clearly unfinished physical line joins the next line", () => {
  const lines = [
    { id: "line_1", words: [word("w1", "hello", 10), word("w2", "my", 80), word("w3", "name", 140), word("w4", "is", 220)] },
    { id: "line_2", words: [word("w5", "Ahmet", 10, 160), word("w6", "im", 90, 160), word("w7", "15", 140, 160), word("w8", "years", 190, 160), word("w9", "old", 270, 160)] }
  ];
  const groups = buildSentenceGroups(lines);
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].lineIds, ["line_1", "line_2"]);
});

test("a complete line stays independent when more lines are added", () => {
  const lines = [
    { id: "line_1", words: [word("w1", "My", 10), word("w2", "favorite", 70), word("w3", "sport", 170), word("w4", "is", 240), word("w5", "football", 280)] },
    { id: "line_2", words: [word("w6", "I", 10, 160), word("w7", "am", 50, 160), word("w8", "15", 100, 160), word("w9", "years", 150, 160), word("w10", "old", 230, 160)] }
  ];
  assert.equal(buildSentenceGroups(lines).length, 2);
});

test("age expressions use a plural unit after numbers other than one", () => {
  const words = [
    word("w1", "I", 10), word("w2", "am", 50), word("w3", "15", 100),
    word("w4", "year", 150), word("w5", "old", 230)
  ];
  const correction = detectDeterministicGrammar(words)
    .find((item) => item.target_id === "w4");
  assert.equal(correction.replacement, "years");
});

test("age expressions tolerate common handwritten OCR confusions", () => {
  const words = [
    word("w1", "I", 10), word("w2", "am", 50), word("w3", "1S", 100),
    word("w4", "ye2r", 150), word("w5", "oll", 230)
  ];
  const correction = detectDeterministicGrammar(words)
    .find((item) => item.target_id === "w4");
  assert.equal(normalizeOcrNumber("1S"), "15");
  assert.equal(correction.replacement, "years");
});

test("mixed letter-number OCR readings accept only nearby word corrections", () => {
  const corrections = [
    { action: "replace", original: "m2tch", replacement: "matches" },
    { action: "replace", original: "m2tch", replacement: "stable" },
    { action: "replace", original: "footbal", replacement: "football" }
  ];
  assert.deepEqual(
    filterLikelyOcrArtifacts(corrections).map((item) => item.replacement),
    ["matches", "football"]
  );
});

test("small printed keyboard labels are removed before sentence grouping", () => {
  const words = [
    { ...word("k1", "R", 20, 20), height: 17 },
    { ...word("k2", "command", 80, 80), height: 18 },
    { ...word("w1", "My", 20, 400), height: 44 },
    { ...word("w2", "name", 100, 400), height: 42 },
    { ...word("w3", "Ahmet", 200, 400), height: 43 }
  ];
  assert.deepEqual(selectHandwrittenWords(words).map((item) => item.id), ["w1", "w2", "w3"]);
});

test("overlapping duplicate OCR readings keep the main word box", () => {
  const words = [
    { ...word("w1", "I", 10, 100), height: 48 },
    { ...word("w2", "year", 100, 100), width: 170, height: 50 },
    { ...word("w3", "scribble", 120, 82), width: 150, height: 30 },
    { ...word("w4", "old", 290, 100), height: 48 }
  ];
  assert.deepEqual(selectHandwrittenWords(words).map((item) => item.id), ["w1", "w2", "w4"]);
});

test("a short perspective line is not removed by a taller neighboring line", () => {
  const words = [
    { id: "upper0", text: "watching", x: 380, y: 300, width: 105, height: 42 },
    { id: "upper", text: "movie", x: 500, y: 300, width: 95, height: 42 },
    { id: "last", text: "Last", x: 500, y: 338, width: 71, height: 12 },
    { id: "week", text: "week", x: 590, y: 338, width: 84, height: 12 }
  ];
  assert.deepEqual(
    selectHandwrittenWords(words).map((item) => item.id).sort(),
    ["last", "upper", "upper0", "week"]
  );
});

test("a proper name in the sentence is never replaced", () => {
  const groups = [{ words: [word("w1", "My", 10), word("w2", "name", 70), word("w3", "is", 150), word("w4", "Arda", 200)] }];
  const corrections = [
    { action: "replace", target_id: "w1", original: "My", replacement: "May" },
    { action: "replace", target_id: "w4", original: "Arda", replacement: "Ardal" }
  ];
  assert.deepEqual(filterProtectedProperNames(corrections, groups).map((item) => item.target_id), ["w1"]);
});

test("a phrase rewrite cannot erase a protected proper name", () => {
  const groups = [{ words: [word("w1", "My", 10), word("w2", "name", 70), word("w3", "is", 150), word("w4", "Arda", 200)] }];
  const corrections = [{
    action: "rewrite_line",
    original: "is Arda",
    replacement: "is Ada",
    target_ids: ["w3", "w4"]
  }];
  assert.deepEqual(filterProtectedProperNames(corrections, groups), []);
});

test("a mid-sentence function word wrongly capitalized is lowercased", () => {
  const group = sentence(["He", "left", "even", "though", "It", "was", "raining"])[0];
  const corrections = detectCapitalizationErrors([group]);
  assert.deepEqual(corrections.map((item) => ({ target_id: item.target_id, replacement: item.replacement })), [
    { target_id: "s4", replacement: "it" }
  ]);
  assert.ok(corrections[0].reason.includes("It"));
});

test("a capitalized proper name mid-sentence is left alone by the capitalization check", () => {
  const group = sentence(["My", "friend", "Arda", "called", "me"])[0];
  const corrections = detectCapitalizationErrors([group]);
  assert.equal(corrections.length, 0);
});

test("a proper name right after a conjunction in an inverted sentence is never flagged", () => {
  const group = sentence(["He", "went", "home", "and", "Ahmet", "stayed", "at", "school"])[0];
  const corrections = detectCapitalizationErrors([group]);
  assert.equal(corrections.some((item) => item.target_id === "s4"), false);
});

test("the pronoun I is never flagged by the capitalization check", () => {
  const group = sentence(["My", "sister", "and", "I", "went", "home"])[0];
  const corrections = detectCapitalizationErrors([group]);
  assert.equal(corrections.length, 0);
});

test("the first word of a sentence group is never flagged by the capitalization check", () => {
  const group = sentence(["The", "dog", "was", "happy"])[0];
  const corrections = detectCapitalizationErrors([group]);
  assert.equal(corrections.length, 0);
});

test("the missing verb after my name survives a curved-line grouping failure", () => {
  const words = [
    { ...word("w1", "My", 100, 110), layoutLine: "split_a" },
    { ...word("w2", "name", 180, 100), layoutLine: "split_a" },
    { ...word("w3", "Ahmet", 330, 82), layoutLine: "split_b" }
  ];
  const correction = detectDeterministicGrammar(words)
    .find((item) => item.action === "insert" && item.replacement === "is");
  assert.equal(correction?.left_id, "w2");
  assert.equal(correction?.right_id, "w3");
});

test("coordinated activities use matching gerund forms", () => {
  const words = [word("w1", "swimming", 10, 100), word("w2", "and", 100, 100), word("w3", "Ply", 160, 180)];
  const corrections = [{ action: "replace", target_id: "w3", original: "Ply", replacement: "play" }];
  assert.equal(enforceParallelCorrectionForms(corrections, [{ words }])[0].replacement, "playing");
});

test("parallel gerund correction does not depend on the model returning an edit", () => {
  const words = [word("w1", "I", 10), word("w2", "like", 50), word("w3", "swimming", 120), word("w4", "and", 220), word("w5", "Ply", 290)];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w5");
  assert.equal(correction.replacement, "playing");
});

test("parallel gerunds may have a noun phrase before the conjunction", () => {
  const words = [
    word("w1", "She", 10), word("w2", "enjoy", 60), word("w3", "reading", 140),
    word("w4", "story", 240), word("w5", "and", 320), word("w6", "play", 380), word("w7", "games", 450)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w6");
  assert.equal(correction.replacement, "playing");
});

test("a third-person singular subject changes dont to doesn't", () => {
  const words = [
    word("w1", "My", 10), word("w2", "brother", 70), word("w3", "dont", 180),
    word("w4", "like", 250), word("w5", "watching", 320)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w3");
  assert.equal(correction.replacement, "doesn't");
});

test("a destination that requires an article receives the", () => {
  const words = [
    word("w1", "My", 10), word("w2", "friend", 60), word("w3", "enjoys", 150),
    word("w4", "going", 240), word("w5", "to", 320), word("w6", "library", 370)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.left_id === "w5" && item.right_id === "w6");
  assert.equal(correction.replacement, "the");
});

test("a plural subject uses the base verb form", () => {
  const words = [word("w1", "they", 10), word("w2", "studies", 90), word("w3", "hard", 190)];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w2");
  assert.equal(correction.replacement, "study");
});

test("multiple objects use a plural object pronoun", () => {
  const words = [
    word("w1", "they", 10), word("w2", "buy", 80), word("w3", "three", 140),
    word("w4", "book", 230), word("w5", "and", 310), word("w6", "read", 370), word("w7", "it", 450)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w7");
  assert.equal(correction.replacement, "them");
});

test("a plural count in one sentence never changes a pronoun in the next sentence", () => {
  const first = sentence(["I", "have", "lived", "here", "for", "three", "years"])[0];
  const second = sentence(["If", "it", "rains", "we", "will", "stay", "home"])[0];
  first.id = "duration";
  second.id = "conditional";
  first.words.forEach((item, index) => { item.id = `duration_${index}`; item.layoutLine = "line_1"; });
  second.words.forEach((item, index) => { item.id = `conditional_${index}`; item.layoutLine = "line_2"; });
  const corrections = detectDeterministicGrammar([...first.words, ...second.words], [first, second]);
  assert.equal(corrections.some((item) => item.original.toLowerCase() === "it" && item.replacement === "them"), false);
});

test("an age expression never changes a later conditional it into them without AI grouping", () => {
  const words = [
    word("w1", "I", 10), word("w2", "am", 70), word("w3", "fifteen", 130),
    word("w4", "years", 240), word("w5", "old", 330), word("w6", "If", 410),
    word("w7", "it", 470), word("w8", "rains", 530)
  ];
  const corrections = detectDeterministicGrammar(words);
  assert.equal(corrections.some((item) => item.target_id === "w7" && item.replacement === "them"), false);
});

test("an age followed by and it never changes it into them", () => {
  const words = [
    word("w1", "I", 10), word("w2", "am", 70), word("w3", "15", 130),
    word("w4", "years", 200), word("w5", "old", 290), word("w6", "and", 360),
    word("w7", "it", 430), word("w8", "is", 490), word("w9", "cold", 540)
  ];
  const corrections = detectDeterministicGrammar(words);
  assert.equal(corrections.some((item) => item.target_id === "w7" && item.replacement === "them"), false);
});

test("a duration never changes a later conditional it into them", () => {
  const words = [
    word("w1", "I", 10), word("w2", "have", 70), word("w3", "lived", 140),
    word("w4", "here", 230), word("w5", "for", 310), word("w6", "three", 380),
    word("w7", "years", 470), word("w8", "If", 10, 180), word("w9", "it", 70, 180),
    word("w10", "rains", 140, 180)
  ];
  const corrections = detectDeterministicGrammar(words);
  assert.equal(corrections.some((item) => item.target_id === "w9" && item.replacement === "them"), false);
});

test("yesterday triggers a nearby irregular past-tense correction", () => {
  const words = [
    word("w1", "She", 10), word("w2", "has", 60), word("w3", "two", 120), word("w4", "cats", 180),
    word("w5", "Yesterday", 250), word("w6", "we", 350), word("w7", "go", 410), word("w8", "to", 460), word("w9", "school", 500)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w7");
  assert.equal(correction.replacement, "went");
});

test("yesterday tolerates a handwritten OCR confusion for go", () => {
  const words = [
    word("w1", "Yesterday", 10), word("w2", "we", 120),
    word("w3", "90", 180), word("w4", "to", 240), word("w5", "school", 290)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w3");
  assert.equal(correction.replacement, "went");
});

test("a general preference pluralizes a nearby singular activity noun", () => {
  const words = [
    word("w1", "I", 10), word("w2", "love", 50), word("w3", "watching", 120),
    word("w4", "football", 230), word("w5", "m2tch", 340)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w5");
  assert.equal(correction.replacement, "matches");
});

test("a third-person preference verb also pluralizes a general count noun", () => {
  const words = [
    word("w1", "She", 10), word("w2", "enjoys", 70), word("w3", "reading", 160),
    word("w4", "story", 260), word("w5", "and", 340), word("w6", "playing", 400)
  ];
  const correction = detectDeterministicGrammar(words).find((item) => item.target_id === "w4");
  assert.equal(correction.replacement, "stories");
});

test("an article keeps a singular activity noun valid", () => {
  const words = [
    word("w1", "I", 10), word("w2", "love", 50), word("w3", "watching", 120),
    word("w4", "a", 230), word("w5", "football", 270), word("w6", "match", 370)
  ];
  assert.equal(detectDeterministicGrammar(words).some((item) => item.target_id === "w6"), false);
});

test("insertions require both neighboring OCR anchors", () => {
  assert.equal(isSafeCorrection({ action: "insert", replacement: "is", left_id: "w1", right_id: "w2" }), true);
  assert.equal(isSafeCorrection({ action: "insert", replacement: "I", left_id: "", right_id: "w1" }), false);
  assert.equal(isSafeCorrection({ action: "insert", replacement: "and", left_id: "w4", right_id: "" }), false);
});

test("validated grammar-word insertions survive before deterministic rules are merged", () => {
  const modelCorrections = [
    { action: "insert", replacement: "am", left_id: "w1", right_id: "w2" },
    { action: "insert", replacement: "a", left_id: "w2", right_id: "w3" },
    { action: "replace", original: "favorit", replacement: "favorite", target_id: "w3" }
  ];
  assert.deepEqual(
    filterLikelyOcrArtifacts(modelCorrections).map((item) => `${item.action}:${item.replacement}`),
    ["insert:am", "insert:a", "replace:favorite"]
  );
});

test("multi-word grammar changes remain one anchored rewrite block", () => {
  const group = sentence(["My", "brother", "doesn't", "enjoy", "to", "read", "books"])[0];
  const corrections = diffLineToCorrections(group, "My brother doesn't enjoy reading books");
  assert.deepEqual(
    corrections.map(({ action, original, replacement, target_ids }) => ({
      action, original, replacement, target_ids
    })),
    [{
      action: "rewrite_line",
      original: "to read",
      replacement: "reading",
      target_ids: ["s4", "s5"]
    }]
  );
});

test("third conditional insertions and irregular participles survive model alignment", () => {
  const group = sentence(["If", "I", "had", "knew", "I", "would", "studied"])[0];
  const corrections = diffLineToCorrections(group, "If I had known I would have studied");
  assert.deepEqual(
    corrections.map((item) => `${item.action}:${item.original || "_"}->${item.replacement}`),
    ["rewrite_line:knew->known", "insert:_->have"]
  );
});

test("suggest complements are rewritten atomically instead of leaving the old to", () => {
  const group = sentence(["She", "suggested", "me", "to", "go", "earlier"])[0];
  const corrections = diffLineToCorrections(group, "She suggested that I go earlier");
  assert.deepEqual(
    corrections.map(({ action, original, replacement, target_ids }) => ({
      action, original, replacement, target_ids
    })),
    [{
      action: "rewrite_line",
      original: "me to",
      replacement: "that I",
      target_ids: ["s2", "s3"]
    }]
  );
});

test("Gemini OCR-ID edits are used only when they exactly rebuild the validated sentence", () => {
  const group = sentence(["My", "brother", "enjoy", "to", "read", "books"])[0];
  const edits = [{
    target_ids: ["s2", "s3", "s4"],
    left_id: "",
    right_id: "",
    replacement: "enjoys reading"
  }];
  const corrections = correctionsFromGeminiEdits(group, edits, "My brother enjoys reading books");
  assert.deepEqual(corrections.map(({ action, target_ids, replacement }) => ({ action, target_ids, replacement })), [{
    action: "rewrite_line",
    target_ids: ["s2", "s3", "s4"],
    replacement: "enjoys reading"
  }]);
  assert.equal(correctionsFromGeminiEdits(group, edits, "My brother enjoys books"), null);
});

test("Gemini's own reason for an edit reaches the typed transcript instead of a generic message", () => {
  const group = sentence(["My", "sister", "like", "reading"])[0];
  const edits = [{
    target_ids: ["s2"],
    left_id: "",
    right_id: "",
    replacement: "likes",
    reason: "Use “likes” because “she” is a third-person singular subject."
  }];
  const corrections = correctionsFromGeminiEdits(group, edits, "My sister likes reading");
  const transcript = buildCorrectionTranscript([group], corrections);
  const changedToken = transcript.find((token) => token.changed);
  assert.equal(changedToken.text, "likes");
  assert.equal(changedToken.reason, "Use “likes” because “she” is a third-person singular subject.");
});

test("a single rendered rewrite cannot erase words from two physical lines", () => {
  const words = [
    { ...word("w1", "will", 100, 100), layoutLine: "line_1" },
    { ...word("w2", "rain", 180, 100), layoutLine: "line_1" },
    { ...word("w3", "we", 100, 210), layoutLine: "line_2" },
    { ...word("w4", "stay", 160, 210), layoutLine: "line_2" }
  ];
  const correction = {
    action: "rewrite_line",
    original: "will rain we stay",
    replacement: "rains we will stay",
    target_ids: ["w1", "w2", "w3", "w4"]
  };
  assert.deepEqual(filterUnrenderableCorrections([correction], words), { renderable: [], dropped: 1 });
});

test("only one replacement can target a word and deterministic correction wins", () => {
  const deterministic = { action: "replace", original: "year", replacement: "years", target_id: "w4", left_id: "", right_id: "" };
  const model = { action: "replace", original: "year", replacement: "old", target_id: "w4", left_id: "", right_id: "" };
  assert.deepEqual(mergeCorrections([deterministic], [model]), [deterministic]);
});

test("a model phrase rewrite cannot overwrite a higher-priority anchored correction", () => {
  const deterministic = { action: "replace", original: "knew", replacement: "known", target_id: "w4", left_id: "", right_id: "" };
  const model = {
    action: "rewrite_line",
    original: "had knew",
    replacement: "had known",
    target_ids: ["w3", "w4"],
    target_id: "", left_id: "", right_id: ""
  };
  assert.deepEqual(mergeCorrections([deterministic], [model]), [deterministic]);
});

test("irregular auxiliary agreement corrections pass the safety filter", () => {
  assert.equal(isSafeCorrection({ action: "replace", original: "was", replacement: "were" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "is", replacement: "are" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "dont", replacement: "doesn't" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "go", replacement: "went" }), true);
});

test("B1-B2 corrections stay inside one logical sentence", () => {
  const groups = [
    ...sentence(["The", "film", "was", "more", "better", "than", "expected"]),
    { ...sentence(["Although", "I", "was", "tired", "but", "I", "finished", "homework"])[0], id: "group_2" },
    { ...sentence(["The", "book", "who", "I", "borrowed", "was", "interesting"])[0], id: "group_3" }
  ];
  groups.forEach((group, groupIndex) => group.words.forEach((item, wordIndex) => {
    item.id = `g${groupIndex}_${wordIndex}`;
  }));
  const byOriginal = new Map(detectAdvancedGrammar(groups).map((item) => [item.original, item.replacement]));
  assert.equal(byOriginal.get("more"), "");
  assert.equal(byOriginal.get("but"), "");
  assert.equal(byOriginal.get("who"), "that");
});

test("joined handwriting for and I live is repaired as one phrase", () => {
  const group = sentence(["I", "am", "fifteen", "years", "old", "an", "ilive", "in", "Ankara"])[0];
  const correction = detectAdvancedGrammar([group]).find((item) => item.action === "rewrite_line");
  assert.equal(correction.original, "an ilive");
  assert.equal(correction.replacement, "and I live");
});

test("conditionals and reported speech receive complete high-confidence corrections", () => {
  const groups = [
    ...sentence(["If", "it", "will", "rain", "tomorrow", "we", "stay", "at", "home"]),
    { ...sentence(["If", "I", "knew", "about", "it", "I", "would", "have", "came", "earlier"])[0], id: "group_2" },
    { ...sentence(["She", "told", "me", "that", "she", "has", "finished", "it"])[0], id: "group_3" }
  ];
  groups.forEach((group, groupIndex) => group.words.forEach((item, wordIndex) => {
    item.id = `c${groupIndex}_${wordIndex}`;
  }));
  const corrections = detectAdvancedGrammar(groups);
  const replacements = new Map(corrections.filter((item) => item.action === "replace")
    .map((item) => [item.original, item.replacement]));
  const rewrites = new Map(corrections.filter((item) => item.action === "rewrite_line")
    .map((item) => [item.original, item.replacement]));
  assert.equal(rewrites.get("will rain"), "rains");
  assert.equal(rewrites.get("knew about"), "had known about");
  assert.equal(replacements.get("came"), "come");
  assert.equal(replacements.get("has"), "had");
  assert.equal(rewrites.get("we stay"), "we will stay");
});

test("a plural subject followed by has and an adjective uses be", () => {
  const corrections = detectAdvancedGrammar(sentence(["We", "has", "happy", "because", "weather", "were", "beautiful"]));
  assert.equal(corrections.find((item) => item.original === "has")?.replacement, "were");
});
