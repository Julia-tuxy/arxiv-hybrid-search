import { useState } from "react";

export default function SearchBar({ onSearch, isLoading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim().length >= 2) onSearch(query.trim());
  };

  return (
    <form onSubmit={handleSubmit} style={{ display: "flex", gap: "8px" }}>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search academic papers…"
        disabled={isLoading}
        style={{
          flex: 1,
          padding: "12px 16px",
          fontSize: "16px",
          border: "1px solid #ccc",
          borderRadius: "6px",
          outline: "none",
        }}
      />
      <button
        type="submit"
        disabled={isLoading || query.trim().length < 2}
        style={{
          padding: "12px 24px",
          fontSize: "16px",
          backgroundColor: "#2563eb",
          color: "#fff",
          border: "none",
          borderRadius: "6px",
          cursor: isLoading || query.trim().length < 2 ? "not-allowed" : "pointer",
          opacity: isLoading || query.trim().length < 2 ? 0.6 : 1,
        }}
      >
        {isLoading ? "Searching…" : "Search"}
      </button>
    </form>
  );
}
