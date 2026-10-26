import { useState } from "react";
import SearchBar from "./components/SearchBar";
import ResultCard from "./components/ResultCard";
import LoadingSpinner from "./components/LoadingSpinner";
import { searchPapers } from "./api/search";

export default function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastQuery, setLastQuery] = useState("");

  const handleSearch = async (query) => {
    setIsLoading(true);
    setError(null);
    setLastQuery(query);
    try {
      const data = await searchPapers(query, 10);
      setResults(data.results);
    } catch {
      setError("Search failed — is the backend running on port 8000?");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: "820px",
        margin: "0 auto",
        padding: "48px 24px",
        fontFamily: "system-ui, -apple-system, sans-serif",
        backgroundColor: "#f8fafc",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ margin: "0 0 4px", fontSize: "28px", fontWeight: 700, color: "#0f172a" }}>
        ArXiv Semantic Search
      </h1>
      <p style={{ margin: "0 0 28px", color: "#64748b", fontSize: "15px" }}>
        Searching ~100K academic papers · SBERT dense retrieval + BM25, fused
      </p>

      <SearchBar onSearch={handleSearch} isLoading={isLoading} />

      {isLoading && <LoadingSpinner />}

      {error && (
        <p style={{ color: "#dc2626", marginTop: "20px", fontSize: "14px" }}>{error}</p>
      )}

      {!isLoading && results.length > 0 && (
        <div style={{ marginTop: "28px" }}>
          <p style={{ color: "#64748b", fontSize: "14px", marginBottom: "16px" }}>
            {results.length} results for &ldquo;{lastQuery}&rdquo;
          </p>
          {results.map((r, i) => (
            <ResultCard key={r.arxiv_id} result={r} rank={i + 1} />
          ))}
        </div>
      )}

      {!isLoading && lastQuery && results.length === 0 && !error && (
        <p style={{ marginTop: "28px", color: "#64748b" }}>
          No results found for &ldquo;{lastQuery}&rdquo;.
        </p>
      )}
    </div>
  );
}
