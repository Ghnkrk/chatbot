import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json

# Pages to scrape (expand if needed)
urls = [
    "https://kct.ac.in/",
    "https://kct.ac.in/kct-placements/",
    "https://kct.ac.in/kct-ceed/",
    "https://kct.ac.in/about/leadership/",
    "https://kct.ac.in/contact-us/",
    "https://kct.ac.in/research/kciri/"
]


def fetch_and_parse(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup


def extract_text(soup):
    texts = []
    for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
        text = tag.get_text(strip=True)
        if text:
            texts.append(text)
    return texts

# Main script
enriched_data = []

for url in urls:
    soup = fetch_and_parse(url)
    texts = extract_text(soup)

    # Get section from URL
    parsed = urlparse(url)
    section = parsed.path.strip("/").replace("-", " ").title() or "Homepage"

    # Chunk text into ~5-line blobs
    chunk = []
    for i, line in enumerate(texts):
        chunk.append(line)
        if len(chunk) >= 5 or i == len(texts) - 1:
            enriched_data.append({
                "content": " ".join(chunk),
                "url": url,
                "section": section
            })
            chunk = []

# Save to JSON file
with open("kct_enriched_data.json", "w", encoding="utf-8") as f:
    json.dump(enriched_data, f, indent=2, ensure_ascii=False)

print("✅ Data saved to 'kct_enriched_data.json'")
