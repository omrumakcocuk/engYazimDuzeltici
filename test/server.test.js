const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildSentenceGroups,
  detectDeterministicGrammar,
  enforceParallelCorrectionForms,
  filterLikelyOcrArtifacts,
  filterProtectedProperNames,
  isSafeCorrection,
  mergeCorrections,
  normalizeOcrNumber,
  parseGoogleVisionWords,
  selectGoogleHandwrittenWords,
  selectHandwrittenWords
} = require("../server");

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

test("a proper name in the sentence is never replaced", () => {
  const groups = [{ words: [word("w1", "My", 10), word("w2", "name", 70), word("w3", "is", 150), word("w4", "Arda", 200)] }];
  const corrections = [
    { action: "replace", target_id: "w1", original: "My", replacement: "May" },
    { action: "replace", target_id: "w4", original: "Arda", replacement: "Ardal" }
  ];
  assert.deepEqual(filterProtectedProperNames(corrections, groups).map((item) => item.target_id), ["w1"]);
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

test("model insertions are discarded before deterministic rules are merged", () => {
  const modelCorrections = [
    { action: "insert", replacement: "am", left_id: "w1", right_id: "w2" },
    { action: "insert", replacement: "a", left_id: "w2", right_id: "w3" },
    { action: "replace", original: "favorit", replacement: "favorite", target_id: "w3" }
  ];
  assert.deepEqual(
    filterLikelyOcrArtifacts(modelCorrections).map((item) => `${item.action}:${item.replacement}`),
    ["insert:a", "replace:favorite"]
  );
});

test("only one replacement can target a word and deterministic correction wins", () => {
  const deterministic = { action: "replace", original: "year", replacement: "years", target_id: "w4", left_id: "", right_id: "" };
  const model = { action: "replace", original: "year", replacement: "old", target_id: "w4", left_id: "", right_id: "" };
  assert.deepEqual(mergeCorrections([deterministic], [model]), [deterministic]);
});

test("irregular auxiliary agreement corrections pass the safety filter", () => {
  assert.equal(isSafeCorrection({ action: "replace", original: "was", replacement: "were" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "is", replacement: "are" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "dont", replacement: "doesn't" }), true);
  assert.equal(isSafeCorrection({ action: "replace", original: "go", replacement: "went" }), true);
});
