import { useState } from "react";

export default function ResultCard({ result, rank }) {
  const [expanded, setExpanded] = useState(false);
  const arxivUrl = `https://arxiv.org/abs/${result.arxiv_id}`;

  const badge = (label, value, color) => (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        marginRight: "6px",
        fontSize: "11px",
        borderRadius: "4px",
        backgroundColor: color,
        color: "#fff",
        fontVariantNumeric: "tabular-nums",
      }}
    >
      {label}: {value.toFixed(3)}
    </span>
  );

  return (
    <div
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: "8px",
        padding: "16px 20px",
        marginBottom: "12px",
        backgroundColor: "#fff",
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "6px",
        }}
      >
        <span style={{ color: "#94a3b8", fontSize: "13px", fontWeight: 600 }}>
          #{rank}
        </span>
        <div>
          {badge("Score", result.score, "#2563eb")}
          {badge("Vector", result.vector_score, "#7c3aed")}
          {badge("BM25", result.bm25_score, "#059669")}
        </div>
      </div>

      <h3 style={{ margin: "0 0 6px", fontSize: "16px", lineHeight: "1.4" }}>
        <a
          href={arxivUrl}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: "#1d4ed8", textDecoration: "none" }}
        >
          {result.title}
        </a>
      </h3>

      <p style={{ margin: "0 0 8px", fontSize: "13px", color: "#64748b" }}>
        {result.authors || "Unknown authors"} &mdash; {result.year || "?"} &mdash;{" "}
        <em>{result.categories}</em>
      </p>

      <p style={{ margin: 0, fontSize: "14px", lineHeight: "1.6", color: "#374151" }}>
        {expanded ? result.abstract_snippet : `${result.abstract_snippet.slice(0, 200)}…`}
        {result.abstract_snippet.length > 200 && (
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              marginLeft: "6px",
              background: "none",
              border: "none",
              color: "#2563eb",
              cursor: "pointer",
              fontSize: "13px",
              padding: 0,
            }}
          >
            {expanded ? "Show less" : "Show more"}
          </button>
        )}
      </p>
    </div>
  );
}
