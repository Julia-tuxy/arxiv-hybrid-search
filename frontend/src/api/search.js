import axios from "axios";

const BASE_URL = "http://localhost:8000";

export async function searchPapers(query, topK = 10) {
  const response = await axios.get(`${BASE_URL}/search`, {
    params: { q: query, top_k: topK },
  });
  return response.data;
}
